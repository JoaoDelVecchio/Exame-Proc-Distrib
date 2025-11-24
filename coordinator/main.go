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
	GenerationsPerCycle = 5 // Quantas gerações rodar antes de migrar
	TotalCycles         = 40 // Quantas vezes repetir o processo
)

// Estrutura para receber dados dos migrantes (compatível com o JSON do Python)
type MigrantsPayload struct {
	Genes [][]float64 `json:"genes"`
}

func main() {
	// Pega a lista de ilhas da variável de ambiente (definida no docker-compose)
	islandsEnv := os.Getenv("ISLANDS")
	if islandsEnv == "" {
		log.Fatal("Nenhuma ilha definida na variável ISLANDS")
	}
	islands := strings.Split(islandsEnv, ",")

	log.Printf("Iniciando Coordenador com %d ilhas: %v", len(islands), islands)

	// 1. Inicialização: Manda todas as ilhas carregarem os dados com RETRY
	log.Println("--- Fase 1: Inicializando Ilhas (Aguardando Workers ficarem online) ---")

	for _, island := range islands {
		connected := false
		// Tenta conectar infinitamente até conseguir
		for !connected {
			resp, err := http.Post(island+"/init", "application/json", nil)

			if err == nil && resp.StatusCode == 200 {
				resp.Body.Close()
				log.Printf("%s conectada e inicializada com sucesso!", island)
				connected = true // Sai do loop de tentativas
			} else {
				// Se der erro, apenas avisa e espera
				log.Printf("%s ainda indisponível (connection refused). Tentando em 2s...", island)
				time.Sleep(2 * time.Second)
			}
		}
	}
	log.Println("Todas as ilhas inicializadas e com dados carregados.")

	// Loop Principal de Evolução
	for cycle := 1; cycle <= TotalCycles; cycle++ {
		log.Printf("\n=== Ciclo %d/%d ===", cycle, TotalCycles)

		// 2. Evolução Paralela
		// Usamos WaitGroup para esperar todas as ilhas terminarem de processar
		var wg sync.WaitGroup
		start := time.Now()

		for _, island := range islands {
			wg.Add(1)
			go func(url string) {
				defer wg.Done()
				// Chama o endpoint /evolve?generations=10
				target := fmt.Sprintf("%s/evolve?generations=%d", url, GenerationsPerCycle)
				resp, err := http.Post(target, "application/json", nil)
				if err != nil {
					log.Printf("Erro evoluindo ilha %s: %v", url, err)
					return
				}
				defer resp.Body.Close()

				// Opcional: Ler o melhor resultado atual
				body, _ := io.ReadAll(resp.Body)
				log.Printf("Ilha %s completou evolução: %s", url, string(body))
			}(island)
		}
		wg.Wait()
		log.Printf("Evolução concluída em %v", time.Since(start))

		// 3. Migração (Topologia Anel)
		// Ilha[0] -> Ilha[1], Ilha[1] -> Ilha[2] ...
		log.Println("--- Iniciando Migração (Topologia Anel) ---")
		for i, islandUrl := range islands {
			// Define quem é o vizinho (o próximo da lista, o último manda pro primeiro)
			nextIndex := (i + 1) % len(islands)
			nextIslandUrl := islands[nextIndex]

			// Passo A: Pega os melhores da ilha atual
			resp, err := http.Get(islandUrl + "/migrants")
			if err != nil {
				log.Printf("Erro ao pegar migrantes de %s: %v", islandUrl, err)
				continue
			}

			migrantsBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			// Passo B: Envia para a próxima ilha
			respSend, err := http.Post(nextIslandUrl+"/migrants", "application/json", bytes.NewBuffer(migrantsBody))
			if err != nil {
				log.Printf("Erro ao enviar migrantes para %s: %v", nextIslandUrl, err)
				continue
			}
			respSend.Body.Close()

			log.Printf("Migração: %s >>> %s (OK)", islandUrl, nextIslandUrl)
		}
	}

	// 4. Resultado Final
	log.Println("\n=== Processo Finalizado. Coletando Resultados ===")
	bestSharpe := -1.0
	bestIsland := ""

	for _, island := range islands {
		resp, err := http.Get(island + "/status")
		if err != nil {
			continue
		}
		var status struct {
			Sharpe float64 `json:"sharpe"`
		}
		json.NewDecoder(resp.Body).Decode(&status)
		resp.Body.Close()

		log.Printf("Resultado Final %s: Sharpe Ratio = %.4f", island, status.Sharpe)

		if status.Sharpe > bestSharpe {
			bestSharpe = status.Sharpe
			bestIsland = island
		}
	}

	log.Printf("\nMELHOR RESULTADO GLOBAL: Sharpe %.4f (Vindo de %s)", bestSharpe, bestIsland)
}
