package naivebayes

import "math"

type NaiveBayes struct {
	classes map[int][][]float64
	means   map[int][]float64
	stddevs map[int][]float64
}

// Gaussian Probability Density Function.
func gaussianPDF(x, mean, std float64) float64 {
	return (1 / (std * math.Sqrt(2*math.Pi))) * math.Exp(-math.Pow(x-mean, 2)/(2*math.Pow(std, 2)))
}

// Calculate mean values for each feature
func calculateMeans(features [][]float64) []float64 {
	featureLength := len(features[0])
	means := make([]float64, featureLength)
	for i := 0; i < featureLength; i++ {
		var sum float64
		for _, feature := range features {
			sum += feature[i]
		}
		means[i] = sum / float64(len(features))
	}
	return means
}

// Calculate standard deviations for each feature
func calculateStddevs(features [][]float64, means []float64) []float64 {
	featureLength := len(features[0])
	stddevs := make([]float64, featureLength)
	for i := 0; i < featureLength; i++ {
		var sum float64
		for _, feature := range features {
			sum += math.Pow(feature[i]-means[i], 2)
		}
		stddevs[i] = math.Sqrt(sum / float64(len(features)))
	}
	return stddevs
}

func (nb *NaiveBayes) Fit(X [][]float64, y []int) {
	// Divide the dataset into categories
	categories := make(map[int][][]float64)
	for i, feature := range X {
		class := y[i]
		categories[class] = append(categories[class], feature)
	}

	// Calculate mean and standard deviation
	means := make(map[int][]float64)
	stddevs := make(map[int][]float64)
	for class, features := range categories {
		means[class] = calculateMeans(features)
		stddevs[class] = calculateStddevs(features, means[class])
	}

	nb.classes = categories
	nb.stddevs = stddevs
	nb.means = means
}

func (nb *NaiveBayes) Predict(X [][]float64) []int {
	predictions := make([]int, len(X))
	for i, features := range X {
		probabilities := make(map[int]float64)
		for class := range nb.classes {
			probabilities[class] = 1.0
			for j, feature := range features {
				// Calculate the probability of the feature belonging to the class using the Gaussian PDF
				probabilities[class] += math.Log(gaussianPDF(feature, nb.means[class][j], nb.stddevs[class][j]))
			}
		}

		// Find the class with the highest probability
		var bestClass int
		max := math.Inf(-1)
		for class, probability := range probabilities {
			if probability > max {
				max = probability
				bestClass = class
			}
		}
		predictions[i] = bestClass
	}
	return predictions
}

func (nb *NaiveBayes) AccuracyScore(preds []int, y []int) float64 {
	var correct float64 = float64(len(preds))
	for i, pred := range preds {
		if pred != y[i] {
			correct--
		}
	}
	accuracy := correct / float64(len(preds))
	return accuracy
}
