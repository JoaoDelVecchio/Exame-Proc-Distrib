# Distributed Genetic Algorithm for Portfolio Optimization

Este projeto foi desenvolvido como **exame final da disciplina CSC-27 — Programação Distribuída (ITA)**.  
O objetivo é aplicar conceitos de **Sistemas Distribuídos** na otimização de portfólios financeiros utilizando **Algoritmos Genéticos** em um modelo distribuído.

---

## Objetivo Geral

Aumentar desempenho, escalabilidade e precisão na busca por portfólios com melhor relação **risco vs retorno**, explorando o paralelismo natural do Algoritmo Genético (GA).

---

## Arquitetura — Modelo de Ilhas

Cada instância do serviço representa uma **ilha**, responsável por evoluir uma subpopulação.  
Em intervalos pré-definidos, as melhores soluções são **migradas** entre as ilhas para manter diversidade e evitar ótimos locais.

Comunicação e sincronização via **HTTP REST**.

---

## ▶️ Como Executar

Com **Docker Desktop** instalado e rodando no Windows/Linux/macOS:

```bash
docker compose up --build
