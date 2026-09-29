package bpe

import (
	_ "embed"
	"encoding/json"
	"github.com/dlclark/regexp2"
	"golang.org/x/text/unicode/norm"
)

//go:embed tokenizer.json
var tokenizerJSON []byte

const unkID = 50280

type Tokenizer struct {
	vocab map[string]int
	ranks map[[2]string]int
	bytes map[byte]rune
	re    *regexp2.Regexp
}

// Load returns the embedded tokenizer (ModernBERT / laya).
func Load() (*Tokenizer, error) {
	return Parse(tokenizerJSON)
}

// Parse reads a Hugging Face tokenizer.json (a model folder's own tokenizer).
func Parse(tokenizerJSON []byte) (*Tokenizer, error) {
	var raw struct {
		Model struct {
			Vocab  map[string]int `json:"vocab"`
			Merges [][]string     `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(tokenizerJSON, &raw); err != nil {
		return nil, err
	}
	ranks := make(map[[2]string]int, len(raw.Model.Merges))
	for i, pair := range raw.Model.Merges {
		if len(pair) == 2 {
			ranks[[2]string{pair[0], pair[1]}] = i
		}
	}
	return &Tokenizer{
		vocab: raw.Model.Vocab,
		ranks: ranks,
		bytes: bytesToUnicode(),
		re:    regexp2.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, regexp2.None),
	}, nil
}

func (t *Tokenizer) Encode(text string) []int {
	text = norm.NFC.String(text)
	var ids []int
	m, _ := t.re.FindStringMatch(text)
	for m != nil {
		piece := m.String()
		var chars []string
		for i := 0; i < len(piece); i++ {
			chars = append(chars, string(t.bytes[piece[i]]))
		}
		for _, tok := range bpe(chars, t.ranks) {
			if id, ok := t.vocab[tok]; ok {
				ids = append(ids, id)
			} else {
				ids = append(ids, unkID)
			}
		}
		m, _ = t.re.FindNextMatch(m)
	}
	return ids
}

func bpe(word []string, ranks map[[2]string]int) []string {
	for len(word) > 1 {
		best, bi := -1, -1
		for i := 0; i < len(word)-1; i++ {
			rank, ok := ranks[[2]string{word[i], word[i+1]}]
			if ok && (best < 0 || rank < best) {
				best, bi = rank, i
			}
		}
		if bi < 0 {
			break
		}
		first, second := word[bi], word[bi+1]
		next := make([]string, 0, len(word)-1)
		for i := 0; i < len(word); {
			if i < len(word)-1 && word[i] == first && word[i+1] == second {
				next = append(next, first+second)
				i += 2
				continue
			}
			next = append(next, word[i])
			i++
		}
		word = next
	}
	return word
}

func bytesToUnicode() map[byte]rune {
	var bs []int
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	have := map[int]bool{}
	for _, b := range bs {
		have[b] = true
	}
	cs := append([]int{}, bs...)
	n := 0
	for b := 0; b < 256; b++ {
		if !have[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	out := make(map[byte]rune, 256)
	for i := range bs {
		out[byte(bs[i])] = rune(cs[i])
	}
	return out
}
