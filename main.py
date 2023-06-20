from Genetic_Algorithm import *
import random

def expected_output(inputs):
    if (inputs[0] == 0.0 and inputs[1] == 0.0) or (inputs[0] == 1.0 and inputs[1] == 1.0):
        return 0.0
    else:
        return 1.0

def main():
    ga = Genetic_Algorithm(100, [0.05, 0.03, 0.8], 2, 1, 1.0, 1.0, 0.4, 1.0)
    inputs = []
    outputs = []
    differences = []

    for l in range(1000):
        inputs.clear()
        outputs.clear()
        differences.clear()

        for k in range(4):
            for i in range(len(ga.population)):
                input = [int(random.uniform(0, 2)), int(random.uniform(0, 2))]
                inputs.append(input)

            outputs = ga.run_population(inputs)

            for j in range(len(outputs)):
                inputs_pair = inputs[j]
                output = outputs[j][0]
                best_output = expected_output(inputs_pair)
                difference = abs(output - best_output)
                if len(differences) != len(outputs):
                    differences.append(-difference)
                else:
                    differences[j] += difference

        ga.set_fitnesses(differences)
        print(ga.current_generation)
        best = ga.get_best_network()
        print("Best fitness: ", best.fitness)
        if best.fitness < 2.0:
            best_inputs = [0.0, 0.0]
            output = best.run(best_inputs)[0]  
            print(output)
            best_output = expected_output(best_inputs)
            print(abs(output - best_output))
            best_inputs = [1.0, 0.0]
            output = best.run(best_inputs)[0]
            print(output)
            best_output = expected_output(best_inputs)
            print(abs(output - best_output))
            best_inputs = [0.0, 1.0]
            output = best.run(best_inputs)[0]
            print(output)
            best_output = expected_output(best_inputs)
            print(abs(output - best_output))
            best_inputs = [1.0, 1.0]
            output = best.run(best_inputs)[0]
            print(output)
            best_output = expected_output(best_inputs)
            print(abs(output - best_output))
        
        ga.create_next_generation()

if __name__ == "__main__":
    main()