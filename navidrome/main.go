// Echora is a thin Navidrome adapter. All analysis and account policy live in Echora.
package main

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/navidrome/navidrome/plugins/pdk/go/host"
	"github.com/navidrome/navidrome/plugins/pdk/go/lyrics"
	"github.com/navidrome/navidrome/plugins/pdk/go/metadata"
	"github.com/navidrome/navidrome/plugins/pdk/go/pdk"
	"github.com/navidrome/navidrome/plugins/pdk/go/sonicsimilarity"
)

var getConfig = pdk.GetConfig
var sendHTTP = host.HTTPSend

type echoraPlugin struct{}

var _ sonicsimilarity.SonicSimilarity = (*echoraPlugin)(nil)
var _ metadata.SimilarSongsByTrackProvider = (*echoraPlugin)(nil)
var _ metadata.SimilarSongsByArtistProvider = (*echoraPlugin)(nil)
var _ metadata.SimilarArtistsProvider = (*echoraPlugin)(nil)
var _ lyrics.Lyrics = (*echoraPlugin)(nil)

func init() {
	plugin := &echoraPlugin{}
	sonicsimilarity.Register(plugin)
	metadata.Register(plugin)
	lyrics.Register(plugin)
}

func boundedCount(count int32, fallback, maximum int32) int32 {
	if count <= 0 {
		return fallback
	}
	if count > maximum {
		return maximum
	}
	return count
}

func callAPI(path string, body, output any) error {
	base, _ := getConfig("apiUrl")
	token, _ := getConfig("apiKey")
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	parsed, err := url.Parse(base)
	if err != nil || parsed == nil || parsed.Host == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("configure a valid Echora plugin API URL")
	}
	if strings.TrimSpace(token) == "" {
		return fmt.Errorf("configure an Echora API key")
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	response, err := sendHTTP(host.HTTPRequest{
		Method: "POST", URL: base + path, Body: payload,
		Headers:           map[string]string{"Authorization": "Bearer " + token, "Content-Type": "application/json"},
		NoFollowRedirects: true, TimeoutMs: 30000,
	})
	// Do not echo host errors: they may contain URL/configuration details.
	if err != nil {
		return fmt.Errorf("Echora request failed; check connectivity and plugin permissions")
	}
	if response == nil {
		return fmt.Errorf("Echora returned no response")
	}
	if response.StatusCode != 200 {
		return fmt.Errorf("Echora returned HTTP %d; check the key, integration settings and analysis coverage", response.StatusCode)
	}
	if err := json.Unmarshal(response.Body, output); err != nil {
		return fmt.Errorf("Echora returned an invalid plugin response")
	}
	return nil
}

func (p *echoraPlugin) GetSonicSimilarTracks(input sonicsimilarity.GetSonicSimilarTracksRequest) (sonicsimilarity.SonicSimilarityResponse, error) {
	result := sonicsimilarity.SonicSimilarityResponse{Matches: []sonicsimilarity.SonicMatch{}}
	input.Count = boundedCount(input.Count, 20, 500)
	err := callAPI("/similar-tracks", input, &result)
	return result, err
}

func (p *echoraPlugin) FindSonicPath(input sonicsimilarity.FindSonicPathRequest) (sonicsimilarity.SonicSimilarityResponse, error) {
	result := sonicsimilarity.SonicSimilarityResponse{Matches: []sonicsimilarity.SonicMatch{}}
	input.Count = boundedCount(input.Count, 25, 500)
	if input.Count < 2 {
		input.Count = 2
	}
	err := callAPI("/sonic-path", input, &result)
	return result, err
}

func (p *echoraPlugin) GetSimilarSongsByTrack(input metadata.SimilarSongsByTrackRequest) (*metadata.SimilarSongsResponse, error) {
	body := map[string]any{"song": map[string]string{"id": input.ID}, "count": boundedCount(input.Count, 20, 500)}
	response := sonicsimilarity.SonicSimilarityResponse{}
	if err := callAPI("/similar-tracks", body, &response); err != nil {
		return nil, err
	}
	result := &metadata.SimilarSongsResponse{Songs: []metadata.SongRef{}}
	for _, match := range response.Matches {
		result.Songs = append(result.Songs, match.Song)
	}
	return result, nil
}

func (p *echoraPlugin) GetSimilarSongsByArtist(input metadata.SimilarSongsByArtistRequest) (*metadata.SimilarSongsResponse, error) {
	result := &metadata.SimilarSongsResponse{Songs: []metadata.SongRef{}}
	input.Count = boundedCount(input.Count, 20, 500)
	err := callAPI("/artist-radio", input, result)
	return result, err
}

func (p *echoraPlugin) GetSimilarArtists(input metadata.SimilarArtistsRequest) (*metadata.SimilarArtistsResponse, error) {
	result := &metadata.SimilarArtistsResponse{Artists: []metadata.ArtistRef{}}
	input.Limit = boundedCount(input.Limit, 10, 100)
	err := callAPI("/similar-artists", input, result)
	return result, err
}

func (p *echoraPlugin) GetLyrics(input lyrics.GetLyricsRequest) (lyrics.GetLyricsResponse, error) {
	result := lyrics.GetLyricsResponse{Lyrics: []lyrics.LyricsText{}}
	err := callAPI("/lyrics", input, &result)
	return result, err
}

func main() {}
