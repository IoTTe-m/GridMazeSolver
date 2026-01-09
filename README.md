# GridMazeSolver
The following project will use EA 🧬 to solve mazes :) 

Work split:
| Piotr Kubala                 | Rafał Łukosz                    | Jakub Kot                        |
|------------------------------|---------------------------------|----------------------------------|
| Algorithmic methods research | Visualization and project setup | Chromosome representation design |
| Testing                      | Code cleanup                    | Introducing DEAP to the code     |
| Implementation               | Implementation                  | Implementation                   |

Data representation:
- The maze will be represented as a rectangular grid filled with 1s and 0s (1 is a wall, 0 is a path). We want to create our own random maze generator instead of using a predefined set of instances.
- The entrance to the maze will always be located in the top left corner, and the exit will be in the bottom right corner.
- To allow for multiple solutions (some more and some less optimal) we will remove a specified percentage of the walls randomly.
- We will represent chromosomes of the agents as a set of numbers that will encode an automaton. We have decided that simple representations would make it too hard for the agents to make big breakthroughs in their pathfinding and they will generalize very poorly.
  
<img width="1487" height="814" alt="Screenshot 2026-01-09 at 10 18 01" src="https://github.com/user-attachments/assets/e1a4129a-09bf-49ff-a35d-05d2f59bd647" />

