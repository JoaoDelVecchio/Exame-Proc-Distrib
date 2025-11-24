package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Configurações
const (
	GenerationsPerCycle = 30    // 30 gerações por ciclo (como você pediu)
	MaxCycles           = 100   // Limite de segurança para não rodar para sempre
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

	// --- FASE 1: INICIALIZAÇÃO ---
	log.Println("--- Fase 1: Inicializando Ilhas ---")
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

	// Variáveis de Controle
	globalBestSharpe := -1.0 // Começa baixo
	startTime := time.Now()  // INICIA O CRONÔMETRO

	// --- FASE 2: LOOP DE EVOLUÇÃO ---
	log.Println("\n--- Iniciando Otimização Distribuída ---")

	for cycle := 1; cycle <= MaxCycles; cycle++ {
		log.Printf("\n=== Ciclo %d (Gerações %d a %d) ===", cycle, (cycle-1)*GenerationsPerCycle, cycle*GenerationsPerCycle)

		// Variável para achar o melhor Sharpe deste ciclo específico
		var cycleBestSharpe float64 = -1.0
		var mu sync.Mutex // Para proteger a escrita da variável acima nas goroutines
		var wg sync.WaitGroup

		// A. Evolução em Paralelo
		for _, island := range islands {
			wg.Add(1)
			go func(url string) {
				defer wg.Done()
				
				// Chama API: /evolve?generations=30
				target := fmt.Sprintf("%s/evolve?generations=%d", url, GenerationsPerCycle)
				resp, err := http.Post(target, "application/json", nil)
				if err != nil {
					log.Printf("Erro evoluindo %s: %v", url, err)
					return
				}
				defer resp.Body.Close()

				// Lê a resposta JSON para saber o Sharpe atual da ilha
				var result EvolveResponse
				if err := json.NewDecoder(resp.Body).Decode(&result); err == nil {
					log.Printf("Ilha %s terminou. Sharpe: %.5f", url, result.CurrentBestSharpe)
					
					// Atualiza o melhor do ciclo de forma segura
					mu.Lock()
					if result.CurrentBestSharpe > cycleBestSharpe {
						cycleBestSharpe = result.CurrentBestSharpe
					}
					mu.Unlock()
				}
			}(island)
		}
		wg.Wait()

		// B. Verificação de Convergência (Critério de Parada)
		improvement := cycleBestSharpe - globalBestSharpe
		log.Printf(">> Melhor Sharpe do Ciclo: %.5f | Melhor Anterior: %.5f | Melhoria: %.5f", cycleBestSharpe, globalBestSharpe, improvement)

		if cycleBestSharpe > globalBestSharpe {
			// Se melhorou, atualizamos o global
			globalBestSharpe = cycleBestSharpe
			
			// Se a melhoria foi insignificante (menor que 0.001), paramos
			// Nota: Só paramos se já tivermos rodado pelo menos 1 ciclo completo antes para comparar
			if cycle > 1 && improvement < ConvergenceTol {
				log.Printf("\nESTAGNAÇÃO DETECTADA: Melhoria %.5f < %.5f. Parando otimização.", improvement, ConvergenceTol)
				break
			}
		} else {
			// Se não melhorou nada (estranho em GA, mas possível), também paramos
			log.Println("\nSEM MELHORIA: Parando otimização.")
			break
		}

		// C. Migração (Topologia Anel) - Só faz se não parou
		log.Println("--- Trocando Indivíduos (Migração) ---")
		for i, islandUrl := range islands {
			nextIndex := (i + 1) % len(islands)
			nextIslandUrl := islands[nextIndex]

			// 1. Get Migrants
			resp, err := http.Get(islandUrl + "/migrants")
			if err != nil {
				continue
			}
			migrantsBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			// 2. Send Migrants
			http.Post(nextIslandUrl+"/migrants", "application/json", bytes.NewBuffer(migrantsBody))
		}
	}

	// --- FIM ---
	elapsed := time.Since(startTime)
	
	log.Println("\n============================================")
	log.Printf("OTIMIZAÇÃO CONCLUÍDA")
	log.Printf("Tempo Total: %s", elapsed)
	log.Printf("Melhor Sharpe Ratio Encontrado: %.5f", globalBestSharpe)
	log.Println("============================================")
}