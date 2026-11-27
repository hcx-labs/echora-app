# Karaoke stems

Echora can keep each song's vocals and instrumental, so the Stage player can turn the singer down
(or up, or play the vocals alone) with one slider. It is optional and off by default.

## What is stored

Analysis already separates vocals with Mel-Band-Roformer, but only into the preprocessing cache,
which expires. With karaoke stems on, a sync also stores, for every song with lyrics, one file:

- Four-channel Opus in WebM. Channels 1 and 2 are the vocals (left, right); channels 3 and 4 are
  the instrumental, which is the decoded song minus the vocals. Together they add back up to the
  song.
- `ECHORA_KARAOKE_STEMS_BITRATE_KBPS` per stereo pair (160 by default, so 320 kb/s per file). At
  that rate Opus is transparent for most listeners, better than 320 kb/s MP3. A four-minute song
  takes about 10 MB; a 10,000-song library about 100 GB.
- The file sits in `ECHORA_KARAOKE_STEMS_DIR` (`/karaoke-stems` in Compose, bound to
  `ECHORA_KARAOKE_STEMS_PATH`, default `./data/karaoke-stems`). **This is durable storage, not a
  cache: put it on a persistent volume** that both the analysis worker (writes) and the backend
  (serves) mount. The worker fails the batch if the directory is not writable.

Changing the bitrate or the separator makes every song's stems again on the next sync. A song
whose stems failed is not tried again until one of them changes.

## Turning it on

Settings → Analysis features → **Karaoke stems**, separately for this server and for Modal. The
next sync separates and encodes the songs that have lyrics and no stems. On Modal the encoded
files come back to the worker; only they are stored here.

## Playing

In the Stage player, the **Karaoke** button opens the stems control when the song has stems:

- Off: the player streams the original song from Navidrome, as always.
- On: it streams the stems file instead, through Web Audio, and the slider mixes the two pairs:
  far left is vocals only, the middle is the full song, far right is the instrumental only.
  Switching keeps the playback position.

Both stems are one stream, so they cannot drift apart. Browsers without four-channel Opus in WebM
do not show the control.
