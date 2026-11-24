package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/navidrome/navidrome/plugins/pdk/go/host"
	"github.com/navidrome/navidrome/plugins/pdk/go/lyrics"
	"github.com/navidrome/navidrome/plugins/pdk/go/metadata"
	"github.com/navidrome/navidrome/plugins/pdk/go/sonicsimilarity"
	"github.com/navidrome/navidrome/plugins/pdk/go/types"
)

func configure(t *testing.T, send func(host.HTTPRequest) (*host.HTTPResponse, error)) {
	t.Helper()
	oldConfig, oldSend := getConfig, sendHTTP
	getConfig = func(key string) (string, bool) {
		if key == "apiUrl" {
			return "https://echora.example/analysis/integrations/navidrome/v1/", true
		}
		return "test-key", true
	}
	sendHTTP = send
	t.Cleanup(func() { getConfig, sendHTTP = oldConfig, oldSend })
}

func TestProviderContracts(t *testing.T) {
	plugin := &echoraPlugin{}
	song := types.SongRef{ID: "navidrome-song", Name: "Seed"}
	cases := []struct {
		name, path, response string
		invoke               func() error
		validate             func(map[string]any) bool
	}{
		{"sonic", "/similar-tracks", `{"matches":[{"song":{"id":"neighbor","name":"Match"},"similarity":0.8}]}`,
			func() error {
				r, e := plugin.GetSonicSimilarTracks(sonicsimilarity.GetSonicSimilarTracksRequest{Song: song})
				if e == nil && (len(r.Matches) != 1 || r.Matches[0].Song.ID != "neighbor") {
					t.Fatal(r)
				}
				return e
			},
			func(b map[string]any) bool {
				return b["song"].(map[string]any)["id"] == song.ID && b["count"] == float64(20)
			}},
		{"path", "/sonic-path", `{"matches":[]}`,
			func() error {
				_, e := plugin.FindSonicPath(sonicsimilarity.FindSonicPathRequest{StartSong: song, EndSong: types.SongRef{ID: "end"}})
				return e
			},
			func(b map[string]any) bool {
				return b["endSong"].(map[string]any)["id"] == "end" && b["count"] == float64(25)
			}},
		{"mix", "/similar-tracks", `{"matches":[{"song":{"id":"neighbor","name":"Match"},"similarity":0.8}]}`,
			func() error {
				r, e := plugin.GetSimilarSongsByTrack(metadata.SimilarSongsByTrackRequest{ID: song.ID, Count: 8})
				if e == nil && (len(r.Songs) != 1 || r.Songs[0].ID != "neighbor") {
					t.Fatal(r)
				}
				return e
			},
			func(b map[string]any) bool {
				return b["song"].(map[string]any)["id"] == song.ID && b["count"] == float64(8)
			}},
		{"radio", "/artist-radio", `{"songs":[]}`,
			func() error {
				_, e := plugin.GetSimilarSongsByArtist(metadata.SimilarSongsByArtistRequest{ID: "artist", Count: 30})
				return e
			},
			func(b map[string]any) bool { return b["id"] == "artist" && b["count"] == float64(30) }},
		{"artists", "/similar-artists", `{"artists":[]}`,
			func() error {
				_, e := plugin.GetSimilarArtists(metadata.SimilarArtistsRequest{ID: "artist", Limit: 7})
				return e
			},
			func(b map[string]any) bool { return b["id"] == "artist" && b["limit"] == float64(7) }},
		{"lyrics", "/lyrics", `{"lyrics":[{"lang":"en","text":"[00:01.000]Hello"}]}`,
			func() error {
				r, e := plugin.GetLyrics(lyrics.GetLyricsRequest{Track: lyrics.TrackInfo{ID: song.ID}})
				if e == nil && len(r.Lyrics) != 1 {
					t.Fatal(r)
				}
				return e
			},
			func(b map[string]any) bool { return b["track"].(map[string]any)["id"] == song.ID }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			configure(t, func(r host.HTTPRequest) (*host.HTTPResponse, error) {
				if r.Method != "POST" || !strings.HasSuffix(r.URL, c.path) || r.Headers["Authorization"] != "Bearer test-key" || !r.NoFollowRedirects || r.TimeoutMs <= 0 {
					t.Fatal(r.Method, r.URL)
				}
				var body map[string]any
				if e := json.Unmarshal(r.Body, &body); e != nil || !c.validate(body) {
					t.Fatal(string(r.Body), e)
				}
				return &host.HTTPResponse{StatusCode: 200, Body: []byte(c.response)}, nil
			})
			if e := c.invoke(); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestFailuresDoNotLeakCredentialsOrRemoteBodies(t *testing.T) {
	for _, status := range []int32{401, 403, 422, 500} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			configure(t, func(host.HTTPRequest) (*host.HTTPResponse, error) {
				return &host.HTTPResponse{StatusCode: status, Body: []byte("test-key private lyrics")}, nil
			})
			var output any
			e := callAPI("/lyrics", map[string]string{}, &output)
			if e == nil || strings.Contains(e.Error(), "test-key") || strings.Contains(e.Error(), "private lyrics") {
				t.Fatal(e)
			}
		})
	}
	configure(t, func(host.HTTPRequest) (*host.HTTPResponse, error) { return nil, errors.New("test-key") })
	var output any
	if e := callAPI("/lyrics", nil, &output); e == nil || strings.Contains(e.Error(), "test-key") {
		t.Fatal(e)
	}
}

func TestSonicPathCountBounds(t *testing.T) {
	for _, tc := range []struct{ input, want int32 }{
		{-1, 25}, {0, 25}, {1, 2}, {2, 2}, {501, 500},
	} {
		t.Run(fmt.Sprint(tc.input), func(t *testing.T) {
			configure(t, func(r host.HTTPRequest) (*host.HTTPResponse, error) {
				var body struct { Count int32 `json:"count"` }
				if err := json.Unmarshal(r.Body, &body); err != nil { t.Fatal(err) }
				if body.Count != tc.want { t.Fatalf("count = %d, want %d", body.Count, tc.want) }
				return &host.HTTPResponse{StatusCode: 200, Body: []byte(`{"matches":[]}`)}, nil
			})
			_, err := (&echoraPlugin{}).FindSonicPath(sonicsimilarity.FindSonicPathRequest{Count: tc.input})
			if err != nil { t.Fatal(err) }
		})
	}
}

func TestInvalidURLNeverSendsKey(t *testing.T) {
	configure(t, func(host.HTTPRequest) (*host.HTTPResponse, error) {
		t.Fatal("unexpected outbound request")
		return nil, nil
	})
	for _, base := range []string{"", "file:///etc/passwd", "https://user:secret@example.com", "https://example.com?token=secret", "https://example.com#fragment"} {
		getConfig = func(key string) (string, bool) {
			if key == "apiUrl" {
				return base, true
			}
			return "test-key", true
		}
		var output any
		if e := callAPI("/lyrics", nil, &output); e == nil {
			t.Fatal(base)
		}
	}
}
