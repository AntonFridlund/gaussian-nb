# Gaussian Naive Bayes
This package includes the Gaussian Naive Bayes algorithm.  
## Install Package
To use the gaussian package install it using the following command:
```bash
go get github.com/antonfridlund/go-gaussian-classifier
```
## Code Example
You can fit the model and run predictions using the following code example:
```go
g := gaussian.NaiveBayes{}
features := [][]float64{
  {0.0, 0.0, 0.0},
  {200.0, 2.0, 200.0},
  {0.1, 0.5, 0.3},
  {201.0, 2.4, 201.0},
}
classes := []int{1, 2, 1, 2}

// Fit the model
g.Fit(features, classes)
// Print predictions
fmt.Println(g.Predict([][]float64{
  {0.02, 0.4, 0.2},
  {200.4, 2.2, 200.3},
  {200.9, 2.1, 200.6},
}))
```
`Output: [1 2 2]`