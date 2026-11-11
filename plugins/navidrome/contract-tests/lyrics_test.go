package contracttests

import (
	"context"
	"os"
	"testing"

	"github.com/navidrome/navidrome/model"
)

func TestEchoraTTML(t *testing.T) {
	contents, err := os.ReadFile("testdata/lyrics.ttml")
	if err != nil {
		t.Fatal(err)
	}
	// Plugins provide raw text without a suffix. Exercise content detection too.
	tracks, err := model.ParseLyrics(context.Background(), "", "en", contents)
	if err != nil {
		t.Fatal(err)
	}
	if len(tracks) != 2 {
		t.Fatalf("expected main and translation, got %+v", tracks)
	}
	main, ok := tracks.Main()
	if !ok || main.Lang != "en" || !main.Synced || len(main.Line) != 2 {
		t.Fatalf("bad main track: %+v", main)
	}
	first := main.Line[0]
	if first.Value != "Hello & world" || *first.Start != 1000 || *first.End != 3000 || len(first.Cue) != 2 {
		t.Fatalf("bad first line: %+v", first)
	}
	if *first.Cue[0].Start != 1000 || *first.Cue[0].End != 2000 || *first.Cue[1].Start != 2000 {
		t.Fatalf("bad syllable timing: %+v", first.Cue)
	}
	if main.Line[1].Value != "こんにちは" {
		t.Fatalf("lost unicode: %+v", main.Line[1])
	}
	translation := tracks[1]
	if translation.Kind != model.LyricKindTranslation || translation.Lang != "es" || len(translation.Line) != 2 {
		t.Fatalf("bad translation: %+v", translation)
	}
	if translation.Line[0].Value != "Hola & mundo" || *translation.Line[0].Start != 1000 || *translation.Line[0].End != 3000 {
		t.Fatalf("bad translated line mapping: %+v", translation.Line[0])
	}
}

// Backing vocals: an inline echo wrapped in a ttm:role="x-bg" span and a whole x-bg line
// parse as cues of a background agent, the lead as the main agent.
func TestEchoraTTMLBackingVocals(t *testing.T) {
	contents, err := os.ReadFile("testdata/lyrics-backing.ttml")
	if err != nil {
		t.Fatal(err)
	}
	tracks, err := model.ParseLyrics(context.Background(), "", "en", contents)
	if err != nil {
		t.Fatal(err)
	}
	main, ok := tracks.Main()
	if !ok || len(main.Line) != 2 {
		t.Fatalf("bad main track: %+v", main)
	}
	roles := map[string]string{}
	for _, agent := range main.Agents {
		roles[agent.ID] = agent.Role
	}
	lead := main.Line[0]
	if lead.Value != "Lonely (so lonely) tonight" || len(lead.Cue) != 6 {
		t.Fatalf("bad lead line: %+v", lead)
	}
	for i, cue := range lead.Cue {
		backing := i >= 2 && i <= 4
		if (roles[cue.AgentID] == "bg") != backing {
			t.Fatalf("cue %d %q: agent %q role %q, backing expected %v", i, cue.Value, cue.AgentID, roles[cue.AgentID], backing)
		}
	}
	if *lead.Cue[2].Start != 1500 || *lead.Cue[4].End != 2400 {
		t.Fatalf("bad echo timing: %+v", lead.Cue[2:5])
	}
	for _, cue := range main.Line[1].Cue {
		if roles[cue.AgentID] != "bg" {
			t.Fatalf("whole backing line not background: %+v", main.Line[1])
		}
	}
}

func TestEchoraLRC(t *testing.T) {
	contents, err := os.ReadFile("testdata/lyrics.lrc")
	if err != nil {
		t.Fatal(err)
	}
	tracks, err := model.ParseLyrics(context.Background(), "", "en", contents)
	if err != nil {
		t.Fatal(err)
	}
	main, ok := tracks.Main()
	if !ok || !main.Synced || len(main.Line) != 2 || main.Line[0].Value != "Hello & world" || *main.Line[0].Start != 1000 || main.Line[1].Value != "こんにちは" {
		t.Fatalf("bad LRC: %+v", tracks)
	}
}

func TestFilteredTranslationKeepsOriginalLineTiming(t *testing.T) {
	contents, err := os.ReadFile("testdata/lyrics-filtered.ttml")
	if err != nil {
		t.Fatal(err)
	}
	tracks, err := model.ParseLyrics(context.Background(), "", "en", contents)
	if err != nil {
		t.Fatal(err)
	}
	if len(tracks) != 2 {
		t.Fatalf("expected main and translation: %+v", tracks)
	}
	main, translation := tracks[0], tracks[1]
	if len(main.Line) != 2 || len(main.Line[0].Cue) != 2 {
		t.Fatalf("main karaoke was altered: %+v", main)
	}
	if translation.Kind != model.LyricKindTranslation || translation.Lang != "es" || len(translation.Line) != 1 {
		t.Fatalf("bad filtered translation: %+v", translation)
	}
	line := translation.Line[0]
	if line.Value != "Hola" || line.Start == nil || *line.Start != 5000 || line.End == nil || *line.End != 6000 {
		t.Fatalf("translation shifted to the wrong main line: %+v", line)
	}
}
