package decide

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/muthuishere/openjevx/internal/bpe"
)

const (
	clsID   = 50281
	sepID   = 50282
	padID   = 50283
	maskID  = 50284
	maxLen  = 1024
	headMax = 256
)

var temperature = [3]float64{1.02553391456604, 1.034013271331787, 1.147902488708496}

type Question struct {
	Type         string
	Instructions string
	Options      []string
	Keys         []string
	QType        int
}

type Item struct {
	IDs     []int
	Markers []int
	QType   int
}

func ParseQuestions(raw map[string]json.RawMessage, order []string) ([]string, []Question, error) {
	ids := order
	if ids == nil {
		for id := range raw {
			ids = append(ids, id)
		}
	}
	out := make([]Question, 0, len(ids))
	for _, id := range ids {
		q, err := parseQuestion(raw[id])
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", id, err)
		}
		out = append(out, q)
	}
	return ids, out, nil
}

func parseQuestion(raw json.RawMessage) (Question, error) {
	var head struct {
		Type         string          `json:"type"`
		Instructions string          `json:"instructions"`
		Criteria     json.RawMessage `json:"criteria"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return Question{}, err
	}
	q := Question{Type: head.Type, Instructions: head.Instructions}
	switch head.Type {
	case "choice":
		q.QType = 0
		keys, vals, err := orderedCriteria(head.Criteria)
		if err != nil {
			return Question{}, err
		}
		q.Keys = keys
		for i, key := range keys {
			if vals[i] == "" {
				q.Options = append(q.Options, key)
			} else {
				q.Options = append(q.Options, key+": "+vals[i])
			}
		}
	case "score":
		q.QType = 1
		var levels []json.RawMessage
		if err := json.Unmarshal(head.Criteria, &levels); err != nil {
			return Question{}, err
		}
		for i, level := range levels {
			q.Keys = append(q.Keys, fmt.Sprint(i))
			q.Options = append(q.Options, fmt.Sprintf("level %d: %s", i, criterion(level)))
		}
	case "noul":
		q.QType = 2
		q.Keys = []string{"false", "true"}
		falseCrit, trueCrit := "no, the statement does not hold", "yes, the statement holds"
		if len(head.Criteria) > 0 && string(head.Criteria) != "null" {
			keys, vals, err := orderedCriteria(head.Criteria)
			if err != nil {
				return Question{}, err
			}
			for i, key := range keys {
				if key == "false" && vals[i] != "" {
					falseCrit = vals[i]
				}
				if key == "true" && vals[i] != "" {
					trueCrit = vals[i]
				}
			}
		}
		q.Options = []string{"false: " + falseCrit, "true: " + trueCrit}
	default:
		return Question{}, fmt.Errorf("unknown type %q", head.Type)
	}
	if len(q.Options) < 2 {
		return Question{}, fmt.Errorf("need at least two options")
	}
	return q, nil
}

func Encode(tok *bpe.Tokenizer, state string, qs []Question) []Item {
	state = strings.ReplaceAll(state, "[MASK]", " ")
	stateIDs := tok.Encode(state)
	items := make([]Item, len(qs))
	for i, q := range qs {
		items[i] = encodeOne(tok, stateIDs, q)
	}
	return items
}

func encodeOne(tok *bpe.Tokenizer, stateIDs []int, q Question) Item {
	ins := strings.ReplaceAll(q.Instructions, "[MASK]", " ")
	head := tok.Encode(q.Type + " question: " + ins)
	optIDs := make([][]int, len(q.Options))
	for i, opt := range q.Options {
		text := " " + strings.ReplaceAll(opt, "[MASK]", " ")
		ids := tok.Encode(text)
		if len(ids) > 48 {
			ids = ids[:48]
		}
		optIDs[i] = append([]int{maskID}, ids...)
	}
	optLen := 0
	for _, ids := range optIDs {
		optLen += len(ids)
	}
	budget := headMax - optLen
	if budget < 16 {
		per := max(4, (headMax-16)/max(1, len(optIDs)))
		for i := range optIDs {
			if len(optIDs[i]) > per {
				optIDs[i] = optIDs[i][:per]
			}
		}
		optLen = 0
		for _, ids := range optIDs {
			optLen += len(ids)
		}
		budget = headMax - optLen
	}
	if keep := max(8, budget); len(head) > keep {
		head = head[:keep]
	}
	ids := append([]int{clsID}, head...)
	ids = append(ids, sepID)
	markers := make([]int, len(optIDs))
	for i, opt := range optIDs {
		markers[i] = len(ids)
		ids = append(ids, opt...)
	}
	ids = append(ids, sepID)
	room := max(0, maxLen-len(ids)-1)
	if len(stateIDs) > room {
		stateIDs = stateIDs[:room]
	}
	ids = append(ids, stateIDs...)
	ids = append(ids, sepID)
	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	kept := markers[:0]
	for _, m := range markers {
		if m < maxLen {
			kept = append(kept, m)
		}
	}
	return Item{IDs: ids, Markers: kept, QType: q.QType}
}

func Decode(ids []string, qs []Question, items []Item, logits []float32, logitWidth int, act []float32) map[string]any {
	out := make(map[string]any, len(ids))
	for i, id := range ids {
		k := len(items[i].Markers)
		scale := temperature[qs[i].QType]
		z := make([]float64, k)
		maxZ := math.Inf(-1)
		for j := 0; j < k; j++ {
			z[j] = float64(logits[i*logitWidth+j]) / scale
			if z[j] > maxZ {
				maxZ = z[j]
			}
		}
		sum := 0.0
		p := make([]float64, k)
		for j := range z {
			p[j] = math.Exp(z[j] - maxZ)
			sum += p[j]
		}
		for j := range p {
			p[j] /= sum
		}
		best := 0
		for j := 1; j < k; j++ {
			if p[j] > p[best] {
				best = j
			}
		}
		probs := map[string]float64{}
		for j, key := range qs[i].Keys {
			probs[key] = round4(p[j])
		}
		conf := round4(p[best])
		ans := map[string]any{
			"type":              qs[i].Type,
			"probabilities":     probs,
			"confidence":        conf,
			"answer_confidence": conf,
		}
		switch qs[i].Type {
		case "choice":
			ans["choice"] = qs[i].Keys[best]
		case "score":
			score := 0.0
			for j := 0; j < k; j++ {
				score += float64(j) * p[j]
			}
			ans["score"] = round4(score)
		default:
			ans["noul"] = round4(p[1])
			ans["confidence"] = round4(math.Max(p[1], 1-p[1]))
		}
		if len(act) >= (i+1)*2 && finite(float64(act[i*2])) && finite(float64(act[i*2+1])) {
			a0, a1 := float64(act[i*2]), float64(act[i*2+1])
			m := math.Max(a0, a1)
			e0, e1 := math.Exp(a0-m), math.Exp(a1-m)
			ans["action"] = map[string]any{"act_probability": round4(e0 / (e0 + e1))}
		}
		out[id] = ans
	}
	return out
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func round4(v float64) float64 {
	if !finite(v) {
		return 0
	}
	return math.Round(v*10000) / 10000
}

func orderedCriteria(raw json.RawMessage) ([]string, []string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil, fmt.Errorf("missing criteria")
	}
	if raw[0] == '[' {
		var vals []json.RawMessage
		if err := json.Unmarshal(raw, &vals); err != nil {
			return nil, nil, err
		}
		keys := make([]string, len(vals))
		out := make([]string, len(vals))
		for i, v := range vals {
			keys[i] = criterion(v)
			out[i] = ""
		}
		return keys, out, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, nil, fmt.Errorf("criteria must be an object or list")
	}
	var keys, vals []string
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return nil, nil, err
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return nil, nil, err
		}
		keys = append(keys, fmt.Sprint(key))
		text := criterion(val)
		if text == "null" {
			text = ""
		}
		vals = append(vals, text)
	}
	return keys, vals, nil
}

func criterion(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return string(raw)
	}
	b, _ := json.Marshal(v)
	return string(b)
}
