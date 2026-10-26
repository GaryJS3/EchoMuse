//go:build server && bench

package main

import (
	"log"
	"os"
	"strings"
	"time"

	"github.com/wilbowes/EchoMuse/internal/bindings/speaker"
)

// Bench only: play a file through the music plane, so a known signal takes
// the firmware's real output path (chain, volume, mixer) while the raw tap
// records. The request file holds the path of raw mono S16 48kHz audio:
//
//	echo /data/local/tmp/x.s16 > /data/local/tmp/em-play-request
const playRequest = "/data/local/tmp/em-play-request"

func init() {
	benchSpeaker = func(spk *speaker.PcmSpeaker) {
		go func() {
			for {
				time.Sleep(500 * time.Millisecond)
				req, err := os.ReadFile(playRequest)
				if err != nil {
					continue
				}
				os.Remove(playRequest)
				path := strings.TrimSpace(string(req))
				pcm, err := os.ReadFile(path)
				if err != nil {
					log.Printf("[benchplay] %v", err)
					continue
				}
				const chunk = 2048 * 2 // one period, mono S16
				log.Printf("[benchplay] playing %s (%.1fs)", path, float64(len(pcm))/96000)
				for off := 0; off+chunk <= len(pcm); off += chunk {
					if err := spk.PumpMusic(pcm[off : off+chunk]); err != nil {
						log.Printf("[benchplay] %v", err)
						break
					}
				}
				spk.EndMusicStream()
				log.Printf("[benchplay] done")
			}
		}()
	}
}
