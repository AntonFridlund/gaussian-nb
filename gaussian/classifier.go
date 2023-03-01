package gaussian

import "math"

type Class int // Used for readability purposes.

type NaiveBayes struct {
	classes map[Class][][]float64
	means   map[Class][]float64
	stddevs map[Class][]float64
}

// Gaussian Probability Density Function.
func gaussianPDF(x, mean, std float64) float64 {
	return (1 / (std * math.Sqrt(2*math.Pi))) * math.Exp(-math.Pow(x-mean, 2)/(2*math.Pow(std, 2)))
}

// Calculate mean values for each feature
func calculateMeans(examples [][]float64) []float64 {
	features := len(examples[0])
	means := make([]float64, features)
	for i := 0; i < features; i++ {
		sum := 0.0
		for _, example := range examples {
			sum += example[i]
		}
		means[i] = sum / float64(len(examples))
	}
	return means
}

// Calculate standard deviations for each feature
func calculateStddevs(examples [][]float64, means []float64) []float64 {
	features := len(examples[0])
	stddevs := make([]float64, features)
	for i := 0; i < features; i++ {
		sum := 0.0
		for _, example := range examples {
			sum += math.Pow(example[i]-means[i], 2)
		}
		stddevs[i] = math.Sqrt(sum / float64(len(examples)))
	}
	return stddevs
}

func (nb *NaiveBayes) Fit(X [][]float64, y []int) {
	// Divide the dataset into categories
	categories := make(map[Class][][]float64)
	for i, example := range X {
		class := Class(y[i])
		categories[class] = append(categories[class], example)
	}

	means := make(map[Class][]float64)
	stddevs := make(map[Class][]float64)
	for class, examples := range categories {
		// Calculate mean values for each feature
		means[class] = calculateMeans(examples)
		// Calculate standard deviations for each feature
		stddevs[class] = calculateStddevs(examples, means[class])
	}

	// Store the data in the NaiveBayes struct
	nb.classes = categories
	nb.stddevs = stddevs
	nb.means = means
}

func (nb *NaiveBayes) Predict(X [][]float64) []int {
	// Slice to store the predictions for each example
	predictions := make([]int, len(X))
	for i, example := range X {
		// Map to store the probability of the example belonging to each class
		probabilities := make(map[Class]float64)
		// Calculate the probability of the example belonging to each class
		for class := range nb.classes {
			probabilities[class] = 1.0
			for j, feature := range example {
				// Calculate the probability of the feature belonging to the class using the Gaussian PDF
				probabilities[class] += math.Log(gaussianPDF(feature, nb.means[class][j], nb.stddevs[class][j]))
			}
		}

		// Find the class with the highest probability
		var bestClass Class
		max := math.Inf(-1)
		for class, probability := range probabilities {
			if probability > max {
				max = probability
				bestClass = class
			}
		}

		// Append the prediction for the example to the slice of predictions
		predictions[i] = int(bestClass)
	}
	return predictions
}

func (nb *NaiveBayes) AccuracyScore(preds []int, y []int) float64 {
	correct := 0
	// Increment correct counter if the prediction is correct
	for i, pred := range preds {
		if pred == y[i] {
			correct++
		}
	}
	// Calculate the accuracy as a percentage and return it
	accuracy := float64(correct) / float64(len(preds))
	return accuracy
}
