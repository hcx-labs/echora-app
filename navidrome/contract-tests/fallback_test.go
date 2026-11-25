package contracttests

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/navidrome/navidrome/conf"
	native "github.com/navidrome/navidrome/core/lyrics"
	"github.com/navidrome/navidrome/model"
)

type pausedLyricsProvider struct{ calls int }

func (p *pausedLyricsProvider) GetLyrics(context.Context, *model.MediaFile) (model.LyricList, error) {
	p.calls++
	return model.LyricList{}, nil
}

func (p *pausedLyricsProvider) LoadLyricsProvider(name string) (native.Provider, bool) {
	return p, name == "echora"
}

func TestPausedPluginFallsThroughToOriginalEmbeddedLyrics(t *testing.T) {
	previous := conf.Server.LyricsPriority
	conf.Server.LyricsPriority = "echora,embedded"
	t.Cleanup(func() { conf.Server.LyricsPriority = previous })
	original := model.LyricList{{Lang: "en", Line: []model.Line{{Value: "Original Navidrome lyrics"}}}}
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	provider := &pausedLyricsProvider{}
	service := native.NewLyrics(nil, provider)
	result, err := service.GetLyrics(context.Background(), &model.MediaFile{Lyrics: string(data)})
	if err != nil {
		t.Fatal(err)
	}
	if provider.calls != 1 || len(result) != 1 || result[0].Line[0].Value != "Original Navidrome lyrics" {
		t.Fatalf("did not fall through to embedded source: calls=%d lyrics=%+v", provider.calls, result)
	}
}
