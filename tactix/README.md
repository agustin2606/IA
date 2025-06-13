# TacTix Environment

## El juego TacTix

### Tablero

TaxTix es jugado en un tablero de N-columnas y N-filas. En este caso, utilizaremos un tablero de 6x6. 

### Piezas

Hay N^2 piezas que cubren todos los espacios del tablero al comenzar el juego. Como se muestra en la imagen:

![alt text](a.svg)

### Reglas

**Movimiento**: En su turno, un jugador puede remover todas las piezas que quiera siempre que (1) esten en la misma fila o columna y (2) esten contiguas.

**Ganador**: En no misére, un jugador gana si logra remover la ultima(s) pieza(s). Es decir, deja al oponente sin piezas para remover.

### Variantes

- **Tamaño del tablero**: El tamaño del tablero puede ser ajustado a cualquier tamaño. Aunque su objetivo va a ser ganar sobre un tablero 6x6, los animamos a probar otros tamaños de tablero.

- **Misére**: En esta variante, el jugador que remueve la última pieza pierde. Es decir, el jugador que deja al oponente sin piezas gana. **No es necesario jugar con esta variante.**

### Ambiente

#### Action-Space

El espacio de acción es un vector de tamaño 4. Donde los elementos son:
- `index`: Indica el indice de la columna o fila donde se van a remover piezas.
- `start`: Indica el indice de la primera pieza que se va a remover.
- `end`: Indica el indice de la última pieza que se va a remover.
- `is_row`: Indica si la acción es sobre una fila o columna. 1 indica que es una fila y 0 indica que es una columna.

## Trainer Agent

El objetivo de este proyecto es crear un agente que pueda competir contra el Trainer Agent. Este agente está preparado para jugar en modo no misére.

Este agente tiene un parámetro que permite ajustar la dificultad del agente. Para eso tenemos un parámetro `difficulty` que controla la probabilidad de que el agente juegue aleatoriamente o de una jugada óptima. Este parámetro puede tomar valores entre 0 y 1. Un valor de 0 significa que el agente siempre juega aleatoriamente, mientras que un valor de 1 significa que el agente siempre juega de manera óptima. 

