package kmeans

import (
	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"code.google.com/p/plotinum/vg"
	"fmt"
	"image/color"
	"io/ioutil"
	"log"
	"math/rand"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Test K-Means Algorithm in Iris Dataset
func TestKmeans(t *testing.T) {
	filePath, err := filepath.Abs("data/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(content), "\n")
	irisData := make([][]float64, len(lines))
	irisLabels := make([]string, len(lines))
	for ii, line := range lines {
		vector := strings.Split(line, ",")
		label := vector[len(vector)-1]
		vector = vector[:len(vector)-1]
		floatVector := make([]float64, len(vector))
		for jj := range vector {
			floatVector[jj], err = strconv.ParseFloat(vector[jj], 64)
		}
		irisData[ii] = floatVector
		irisLabels[ii] = label
	}
	threshold := 10
	// Best Distance for Iris is Canberra Distance
	labels, err := Kmeans(irisData, 3, CanberraDistance, threshold)
	if err != nil {
		log.Fatal(err)
	}

	misclassifiedOnes := 0
	for ii, jj := range labels {
		if ii < 50 {
			if jj != 2 {
				misclassifiedOnes++
			}
		} else if ii < 100 {
			if jj != 1 {
				misclassifiedOnes++
			}
		} else {
			if jj != 0 {
				misclassifiedOnes++
			}
		}
	}
}

func TestKCmeans(t *testing.T) {
	filePath, err := filepath.Abs("data/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(content), "\n")
	irisData := make([][]float64, len(lines))
	irisLabels := make([]string, len(lines))
	for ii, line := range lines {
		vector := strings.Split(line, ",")
		label := vector[len(vector)-1]
		vector = vector[:len(vector)-1]
		floatVector := make([]float64, len(vector))
		for jj := range vector {
			floatVector[jj], err = strconv.ParseFloat(vector[jj], 64)
		}
		irisData[ii] = floatVector
		irisLabels[ii] = label
	}
	threshold := 1000
	// Best Distance for Iris is Canberra Distance
	_, means, err := KCmeans(irisData, len(lines), CanberraDistance, threshold)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("means = %v\n", len(means))
}

const POINTS = 256

func TestKCmeansSynthetic(t *testing.T) {
	clusters := 10
	data, points := make([][]float64, clusters * POINTS), make(plotter.XYs, clusters * POINTS)

	for c := 0; c < clusters; c++ {
		for p := 0; p < POINTS; p++ {
			point := make([]float64, 2)
			point[0], point[1] = 8 * float64(c) + rand.NormFloat64(), 8 * float64(c) + rand.NormFloat64()
			index := POINTS * c + p
			data[index], points[index].X, points[index].Y = point, point[0], point[1]
		}
	}

	threshold := 1000
	_, means, err := KCmeans(data, 30, EuclideanDistance, threshold)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("means = %v\n", len(means))
	centerPoints := make(plotter.XYs, len(means))
	for ii := range means {
		centerPoints[ii].X, centerPoints[ii].Y = means[ii][0], means[ii][1]
	}

	p, err := plot.New()
	p.Title.Text = "Clusters"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	scatter, err := plotter.NewScatter(points)
	scatter.Shape = plot.CircleGlyph{}
	scatter.Radius = vg.Points(2)
	p.Add(scatter)
	scatter, err = plotter.NewScatter(centerPoints)
	scatter.Shape = plot.CircleGlyph{}
	scatter.Radius = vg.Points(4)
	scatter.Color = color.RGBA{0, 0, 255, 255}
	p.Add(scatter)
	if err := p.Save(8, 8, "synthetic.png"); err != nil {
		panic(err)
	}
}

func TestKCmeansA1(t *testing.T) {
	filePath, err := filepath.Abs("data/a1.txt")
	if err != nil {
		log.Fatal(err)
	}
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(content), "\n")
	lines = lines[:len(lines) - 1]
	data := make([][]float64, len(lines))
	for ii, line := range lines {
		line = strings.Trim(line, " ")
		vector := strings.Split(line, " ")
		floatVector := make([]float64, 2)
		vv := 0
		for jj := range vector {
			if vector[jj] == "" || vector[jj] == " " {
				continue
			}
			value, err := strconv.ParseFloat(vector[jj], 64)
			if err != nil {
				log.Fatal(err)
			}
			if value != 0 {
				floatVector[vv] = value
				vv++
			}
		}
		data[ii] = floatVector
	}

	threshold := 1000
	_, means, err := KCmeans(data, 30, EuclideanDistance, threshold)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("means = %v\n", len(means))
}
