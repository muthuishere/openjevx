package bpe

import "testing"

func TestEncodeMatchesTokenizers(t *testing.T) {
	tok, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]int{
		"hello world":                  {25521, 1533},
		"choice question: Which team?": {22122, 1953, 27, 6758, 2285, 32},
		" level 0: low":                {1268, 470, 27, 1698},
		"We were billed twice.":        {1231, 497, 47045, 7019, 15},
	}
	for text, ids := range want {
		got := tok.Encode(text)
		if len(got) != len(ids) {
			t.Fatalf("%q len %d want %d got %v", text, len(got), len(ids), got)
		}
		for i := range ids {
			if got[i] != ids[i] {
				t.Fatalf("%q at %d got %d want %d full %v", text, i, got[i], ids[i], got)
			}
		}
	}
}
