package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Configurações
const (
	GenerationsPerCycle = 30    // 30 gerações por ciclo
	MaxCycles           = 200   // Limite de segurança
	ConvergenceTol      = 0.001 // Tolerância de estagnação
)

// Estruturas JSON para comunicação
type MigrantsPayload struct {
	Genes [][]float64 `json:"genes"`
}

type EvolveResponse struct {
	Status            string  `json:"status"`
	CurrentBestSharpe float64 `json:"current_best_sharpe"`
}

func main() {
	islandsEnv := os.Getenv("ISLANDS")
	if islandsEnv == "" {
		log.Fatal("Nenhuma ilha definida na variável ISLANDS")
	}
	islands := strings.Split(islandsEnv, ",")

	log.Printf("Iniciando Coordenador com %d ilhas: %v", len(islands), islands)

	log.Println("Inicializando Ilhas")
	for _, island := range islands {
		connected := false
		for !connected {
			resp, err := http.Post(island+"/init", "application/json", nil)
			if err == nil && resp.StatusCode == 200 {
				resp.Body.Close()
				log.Printf("%s online!", island)
				connected = true
			} else {
				log.Printf("%s indisponível. Tentando em 2s...", island)
				time.Sleep(2 * time.Second)
			}
		}
	}

	globalBestSharpe := -1.0
	startTime := time.Now()

	log.Println("\nIniciando Otimização Distribuída")

	for cycle := 1; cycle <= MaxCycles; cycle++ {
		log.Printf("\n=== Ciclo %d (Gerações %d a %d) ===", cycle, (cycle-1)*GenerationsPerCycle, cycle*GenerationsPerCycle)

		var cycleBestSharpe float64 = -1.0
		var mu sync.Mutex
		var wg sync.WaitGroup

		for _, island := range islands {
			wg.Add(1)
			go func(url string) {
				defer wg.Done()
				
				target := fmt.Sprintf("%s/evolve?generations=%d", url, GenerationsPerCycle)
				resp, err := http.Post(target, "application/json", nil)
				if err != nil {
					log.Printf("Erro evoluindo %s: %v", url, err)
					return
				}
				defer resp.Body.Close()

				var result EvolveResponse
				if err := json.NewDecoder(resp.Body).Decode(&result); err == nil {
					log.Printf("Ilha %s terminou. Sharpe: %.5f", url, result.CurrentBestSharpe)
					
					mu.Lock()
					if result.CurrentBestSharpe > cycleBestSharpe {
						cycleBestSharpe = result.CurrentBestSharpe
					}
					mu.Unlock()
				}
			}(island)
		}
		wg.Wait()

		improvement := cycleBestSharpe - globalBestSharpe
		log.Printf(">> Melhor Sharpe do Ciclo: %.5f | Melhor Anterior: %.5f | Melhoria: %.5f", cycleBestSharpe, globalBestSharpe, improvement)

		if cycleBestSharpe > globalBestSharpe {
			globalBestSharpe = cycleBestSharpe
			
			if cycle > 1 && improvement < ConvergenceTol {
				log.Printf("\nESTAGNAÇÃO DETECTADA: Melhoria %.5f < %.5f. Parando otimização.", improvement, ConvergenceTol)
				break
			}
		} else {
			log.Println("\nSEM MELHORIA: Parando otimização.")
			break
		}

		log.Println("--- Trocando Indivíduos (Migração) ---")
		for i, islandUrl := range islands {
			nextIndex := (i + 1) % len(islands)
			nextIslandUrl := islands[nextIndex]

			resp, err := http.Get(islandUrl + "/migrants")
			if err != nil {
				continue
			}
			migrantsBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			http.Post(nextIslandUrl+"/migrants", "application/json", bytes.NewBuffer(migrantsBody))
		}
	}
	elapsed := time.Since(startTime)
	
	log.Println("\n============================================")
	log.Printf("OTIMIZAÇÃO CONCLUÍDA")
	log.Printf("Tempo Total: %s", elapsed)
	log.Printf("Melhor Sharpe Ratio Encontrado: %.5f", globalBestSharpe)
	log.Println("============================================")
}