package kmeans

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image/color"
	"math"
	"math/cmplx"
	"math/rand"

	//"code.google.com/p/lzma"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/mjibson/go-dsp/fft"
	"github.com/pointlander/compress"
)

// Observation: Data Abstraction for an N-dimensional
// observation
type Observation []float64

// Abstracts the Observation with a cluster number
// Update and computeation becomes more efficient
type ClusteredObservation struct {
	ClusterNumber int
	Observation
}

// Distance Function: To compute the distanfe between observations
type DistanceFunction func(first, second []float64) (float64, error)

/*
func (observation Observation) Sqd(otherObservation Observation) (ssq float64) {
	for ii, jj := range observation {
		d := jj - otherObservation[ii]
		ssq += d * d
	}
	return ssq
}
*/

// Summation of two vectors
func (observation Observation) Add(otherObservation Observation) {
	for ii, jj := range otherObservation {
		observation[ii] += jj
	}
}

// Multiplication of a vector with a scalar
func (observation Observation) Mul(scalar float64) {
	for ii := range observation {
		observation[ii] *= scalar
	}
}

// Dot Product of Two vectors
func (observation Observation) InnerProduct(otherObservation Observation) {
	for ii := range observation {
		observation[ii] *= otherObservation[ii]
	}
}

// Outer Product of two arrays
// TODO: Need to be tested
func (observation Observation) OuterProduct(otherObservation Observation) [][]float64 {
	result := make([][]float64, len(observation))
	for ii := range result {
		result[ii] = make([]float64, len(otherObservation))
	}
	for ii := range result {
		for jj := range result[ii] {
			result[ii][jj] = observation[ii] * otherObservation[jj]
		}
	}
	return result
}

// Find the closest observation and return the distance
// Index of observation, distance
func near(p ClusteredObservation, mean []Observation, distanceFunction DistanceFunction) (int, float64) {
	indexOfCluster := 0
	minSquaredDistance, _ := distanceFunction(p.Observation, mean[0])
	for i := 1; i < len(mean); i++ {
		squaredDistance, _ := distanceFunction(p.Observation, mean[i])
		if squaredDistance < minSquaredDistance {
			minSquaredDistance = squaredDistance
			indexOfCluster = i
		}
	}
	return indexOfCluster, math.Sqrt(minSquaredDistance)
}

// Instead of initializing randomly the seeds, make a sound decision of initializing
func seed(data []ClusteredObservation, k int, distanceFunction DistanceFunction) []Observation {
	s := make([]Observation, k)
	s[0] = data[rand.Intn(len(data))].Observation
	d2 := make([]float64, len(data))
	for ii := 1; ii < k; ii++ {
		var sum float64
		for jj, p := range data {
			_, dMin := near(p, s[:ii], distanceFunction)
			d2[jj] = dMin * dMin
			sum += d2[jj]
		}
		target := rand.Float64() * sum
		jj := 0
		for sum = d2[0]; sum < target; sum += d2[jj] {
			jj++
		}
		s[ii] = data[jj].Observation
	}
	return s
}

// K-Means Algorithm
func kmeans(data []ClusteredObservation, mean []Observation, distanceFunction DistanceFunction, threshold int) ([]ClusteredObservation, error) {
	counter := 0
	for ii, jj := range data {
		closestCluster, _ := near(jj, mean, distanceFunction)
		data[ii].ClusterNumber = closestCluster
	}
	mLen := make([]int, len(mean))
	for n := len(data[0].Observation); ; {
		for ii := range mean {
			mean[ii] = make(Observation, n)
			mLen[ii] = 0
		}
		for _, p := range data {
			mean[p.ClusterNumber].Add(p.Observation)
			mLen[p.ClusterNumber]++
		}
		for ii := range mean {
			mean[ii].Mul(1 / float64(mLen[ii]))
		}
		var changes int
		for ii, p := range data {
			if closestCluster, _ := near(p, mean, distanceFunction); closestCluster != p.ClusterNumber {
				changes++
				data[ii].ClusterNumber = closestCluster
			}
		}
		counter++
		if changes == 0 || counter > threshold {
			return data, nil
		}
	}
	return data, nil
}

// K-Means Algorithm with smart seeds
// as known as K-Means ++
func Kmeans(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold int) ([]int, []Observation, error) {
	data := make([]ClusteredObservation, len(rawData))
	for ii, jj := range rawData {
		data[ii].Observation = jj
	}
	seeds := seed(data, k, distanceFunction)
	clusteredData, err := kmeans(data, seeds, distanceFunction, threshold)
	labels := make([]int, len(clusteredData))
	for ii, jj := range clusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, seeds, err
}

func KCmeans(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold, gain int) ([]int, []Observation, error) {
	var minClusteredData []ClusteredObservation
	var means []Observation
	var err error
	min := int64(math.MaxInt64)
	for clusters := 1; clusters <= k; clusters++ {
		data := make([]ClusteredObservation, len(rawData))
		for ii, jj := range rawData {
			data[ii].Observation = jj
		}
		seeds := seed(data, clusters, distanceFunction)
		clusteredData, _ := kmeans(data, seeds, distanceFunction, threshold)

		counts := make([]int, clusters)
		for _, jj := range clusteredData {
			counts[jj.ClusterNumber]++
		}

		input := &bytes.Buffer{}
		for c := 0; c < clusters; c++ {
			err := binary.Write(input, binary.LittleEndian, rand.Float64())
			if err != nil {
				panic(err)
			}

			err = binary.Write(input, binary.LittleEndian, int64(counts[c]))
			if err != nil {
				panic(err)
			}

			for _, jj := range seeds[c] {
				err = binary.Write(input, binary.LittleEndian, jj)
				if err != nil {
					panic(err)
				}
			}

			/*sigma := make([]float64, len(seeds[c]))*/
			for _, j := range clusteredData {
				if j.ClusterNumber == c {
					for ii, jj := range j.Observation {
						x := jj - seeds[c][ii]
						//sigma[ii] += x * x
						err = binary.Write(input, binary.LittleEndian, x)
						if err != nil {
							panic(err)
						}
					}

					for ii, jj := range j.Observation {
						x := math.Exp(jj - seeds[c][ii])
						for i := 0; i < gain; i++ {
							err = binary.Write(input, binary.LittleEndian, x*rand.Float64())
							if err != nil {
								panic(err)
							}
						}
					}
				}
			}

			/*N := float64(counts[c])
			for i, j := range sigma {
				sigma[i] = math.Sqrt(j / N)
			}

			for i := 0; i < gain * counts[c]; i++ {
				for _, jj := range sigma {
					err = binary.Write(input, binary.LittleEndian, 3 * jj * rand.NormFloat64())
					if err != nil {
						panic(err)
					}
				}
			}*/
		}

		in, output := make(chan []byte, 1), &bytes.Buffer{}
		in <- input.Bytes()
		close(in)
		compress.BijectiveBurrowsWheelerCoder(in).MoveToFrontRunLengthCoder().AdaptiveCoder().Code(output)

		/*output := &bytes.Buffer{}
		writer := lzma.NewWriterLevel(output, lzma.BestCompression)
		writer.Write(input.Bytes())
		writer.Close()*/

		complexity := int64(output.Len())
		fmt.Printf("%v %v\n", clusters, complexity)
		if complexity < min {
			min, minClusteredData, means = complexity, clusteredData, make([]Observation, len(seeds))
			for ii := range seeds {
				means[ii] = make([]float64, len(seeds[ii]))
				for jj := range seeds[ii] {
					means[ii][jj] = seeds[ii][jj]
				}
			}
		}
	}

	labels := make([]int, len(minClusteredData))
	for ii, jj := range minClusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, means, err
}

func kc(a []byte) float64 {
	input, in, output := make([]byte, len(a)), make(chan []byte, 1), &bytes.Buffer{}
	copy(input, a)
	in <- input
	close(in)
	compress.BijectiveBurrowsWheelerCoder(in).MoveToFrontRunLengthCoder().AdaptiveCoder().Code(output)
	return float64(output.Len())
}

func KC2means(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold, gain int) ([]int, []Observation, error) {
	var minClusteredData []ClusteredObservation
	var means []Observation
	var err error
	min := math.MaxFloat64
	for clusters := 1; clusters <= k; clusters++ {
		data := make([]ClusteredObservation, len(rawData))
		for ii, jj := range rawData {
			data[ii].Observation = jj
		}
		seeds := seed(data, clusters, distanceFunction)
		clusteredData, _ := kmeans(data, seeds, distanceFunction, threshold)

		counts := make([]int, clusters)
		for _, jj := range clusteredData {
			counts[jj.ClusterNumber]++
		}

		input, synth := &bytes.Buffer{}, &bytes.Buffer{}
		for c := 0; c < clusters; c++ {
			/*err := binary.Write(input, binary.LittleEndian, int64(counts[c]))
			if err != nil {
				panic(err)
			}

			for _, jj := range seeds[c] {
				err = binary.Write(input, binary.LittleEndian, jj)
				if err != nil {
					panic(err)
				}
			}

			err = binary.Write(synth, binary.LittleEndian, int64(counts[c]))
			if err != nil {
				panic(err)
			}

			for _, jj := range seeds[c] {
				err = binary.Write(synth, binary.LittleEndian, jj)
				if err != nil {
					panic(err)
				}
			}*/

			sigma := make([]float64, len(seeds[c]))
			for _, j := range clusteredData {
				if j.ClusterNumber == c {
					for ii, jj := range j.Observation {
						x := jj - seeds[c][ii]
						sigma[ii] += x * x
						err := binary.Write(input, binary.LittleEndian, jj)
						if err != nil {
							panic(err)
						}
					}
				}
			}

			N := float64(counts[c])
			for i, j := range sigma {
				sigma[i] = math.Sqrt(j / N)
			}

			for i := 0; i < 2*counts[c]; i++ {
				for ii, jj := range sigma {
					err := binary.Write(synth, binary.LittleEndian, jj*rand.NormFloat64()+seeds[c][ii])
					if err != nil {
						panic(err)
					}
				}
			}
		}

		x, y := kc(input.Bytes()), kc(synth.Bytes())
		input.Write(synth.Bytes())
		xy := kc(input.Bytes())
		NCD := (xy - math.Min(x, y)) / math.Max(x, y)

		fmt.Printf("%v %v\n", clusters, NCD)
		if NCD < min {
			min, minClusteredData, means = NCD, clusteredData, make([]Observation, len(seeds))
			for ii := range seeds {
				means[ii] = make([]float64, len(seeds[ii]))
				for jj := range seeds[ii] {
					means[ii][jj] = seeds[ii][jj]
				}
			}
		}
	}

	labels := make([]int, len(minClusteredData))
	for ii, jj := range minClusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, means, err
}

func KCMmeans(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold int) ([]int, []Observation, error) {
	var minClusteredData []ClusteredObservation
	var means []Observation
	var err error
	max, trace := int64(0), make([]float64, k)
	for clusters := 1; clusters <= k; clusters++ {
		data := make([]ClusteredObservation, len(rawData))
		for ii, jj := range rawData {
			data[ii].Observation = jj
		}
		seeds := seed(data, clusters, distanceFunction)
		clusteredData, _ := kmeans(data, seeds, distanceFunction, threshold)

		counts := make([]int, clusters)
		for _, jj := range clusteredData {
			counts[jj.ClusterNumber]++
		}

		input := &bytes.Buffer{}
		for c := 0; c < clusters; c++ {
			/*err := binary.Write(input, binary.LittleEndian, rand.Float64())
			if err != nil {
				panic(err)
			}*/

			err := binary.Write(input, binary.LittleEndian, int64(counts[c]))
			if err != nil {
				panic(err)
			}

			for _, jj := range seeds[c] {
				err = binary.Write(input, binary.LittleEndian, jj)
				if err != nil {
					panic(err)
				}
			}

			for _, j := range clusteredData {
				if j.ClusterNumber == c {
					for ii, jj := range j.Observation {
						err = binary.Write(input, binary.LittleEndian, jj-seeds[c][ii])
						if err != nil {
							panic(err)
						}
					}
				}
			}
		}

		in, output := make(chan []byte, 1), &bytes.Buffer{}
		in <- input.Bytes()
		close(in)
		compress.BijectiveBurrowsWheelerCoder(in).MoveToFrontRunLengthCoder().AdaptiveCoder().Code(output)

		/*output := &bytes.Buffer{}
		writer := lzma.NewWriterLevel(output, lzma.BestCompression)
		writer.Write(input.Bytes())
		writer.Close()*/

		complexity := int64(output.Len())
		trace[clusters-1] = float64(complexity)
		fmt.Printf("%v %v\n", clusters, complexity)
		if complexity > max {
			max, minClusteredData, means = complexity, clusteredData, make([]Observation, len(seeds))
			for ii := range seeds {
				means[ii] = make([]float64, len(seeds[ii]))
				for jj := range seeds[ii] {
					means[ii][jj] = seeds[ii][jj]
				}
			}
		}
	}

	f := fft.FFTReal(trace)
	points, phase, complex := make(plotter.XYs, len(f)-1), make(plotter.XYs, len(f)-1), make(plotter.XYs, len(f))
	for i, j := range f[1:] {
		points[i].X, points[i].Y = float64(i), cmplx.Abs(j)
		phase[i].X, phase[i].Y = float64(i), cmplx.Phase(j)
		complex[i].X, complex[i].Y = real(j), imag(j)
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "FFT Real"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.Shape = draw.CircleGlyph{}
	scatter.Radius = vg.Points(1)
	p.Add(scatter)
	if err := p.Save(8, 8, "fft_real.png"); err != nil {
		panic(err)
	}

	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "FFT Phase"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	scatter, err = plotter.NewScatter(phase)
	if err != nil {
		panic(err)
	}
	scatter.Shape = draw.CircleGlyph{}
	scatter.Radius = vg.Points(1)
	scatter.Color = color.RGBA{0, 0, 255, 255}
	p.Add(scatter)
	if err := p.Save(8, 8, "fft_phase.png"); err != nil {
		panic(err)
	}

	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "FFT Complex"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	scatter, err = plotter.NewScatter(complex)
	if err != nil {
		panic(err)
	}
	scatter.Shape = draw.CircleGlyph{}
	scatter.Radius = vg.Points(1)
	scatter.Color = color.RGBA{0, 0, 255, 255}
	p.Add(scatter)
	if err := p.Save(8, 8, "fft_complex.png"); err != nil {
		panic(err)
	}

	labels := make([]int, len(minClusteredData))
	for ii, jj := range minClusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, means, err
}

func KCSmeans(rawData [][]float64, k int, distanceFunction DistanceFunction, threshold int) ([]int, error) {
	var clusteredData []ClusteredObservation
	var err error
	for clusters := 1; clusters <= k; clusters++ {
		data := make([]ClusteredObservation, len(rawData))
		for ii, jj := range rawData {
			data[ii].Observation = jj
		}
		seeds := seed(data, clusters, distanceFunction)
		clusteredData, err = kmeans(data, seeds, distanceFunction, threshold)

		counts := make([]int, clusters)
		for _, jj := range clusteredData {
			counts[jj.ClusterNumber]++
		}

		input, width := &bytes.Buffer{}, len(seeds[0])
		x := make([]float64, width)
		for c := 0; c < clusters; c++ {
			err := binary.Write(input, binary.LittleEndian, rand.Float64())
			if err != nil {
				panic(err)
			}
			err = binary.Write(input, binary.LittleEndian, int64(counts[c]))
			if err != nil {
				panic(err)
			}
			for _, jj := range seeds[c] {
				err = binary.Write(input, binary.LittleEndian, jj)
				if err != nil {
					panic(err)
				}
			}
			for _, j := range clusteredData {
				if j.ClusterNumber == c {
					/*distance, _ := distanceFunction(j.Observation, seeds[c])
					err = binary.Write(input, binary.LittleEndian, distance)
					if err != nil {
						panic(err)
					}*/

					for ii, jj := range j.Observation {
						x[ii] = jj - seeds[c][ii]
					}

					if width == 1 {
						err = binary.Write(input, binary.LittleEndian, x[0])
						if err != nil {
							panic(err)
						}
					} else {
						r := 0.0
						for _, i := range x {
							r += i * i
						}
						err = binary.Write(input, binary.LittleEndian, math.Sqrt(r))
						if err != nil {
							panic(err)
						}

						t := math.Acos(x[1] / math.Sqrt(x[0]*x[0]+x[1]*x[1]))
						if t < 0 {
							t = 2*math.Pi - t
						}
						err = binary.Write(input, binary.LittleEndian, t)
						if err != nil {
							panic(err)
						}

						for i := 2; i < width; i++ {
							r = 0.0
							for _, j := range x[:i+1] {
								r += j * j
							}
							err = binary.Write(input, binary.LittleEndian, math.Acos(x[i]/math.Sqrt(r)))
							if err != nil {
								panic(err)
							}
						}
					}
				}
			}
		}

		in, output := make(chan []byte, 1), &bytes.Buffer{}
		in <- input.Bytes()
		close(in)
		compress.BijectiveBurrowsWheelerCoder(in).MoveToFrontRunLengthCoder().AdaptiveCoder().Code(output)
		fmt.Printf("%v %v\n", clusters, output.Len())
	}

	labels := make([]int, len(clusteredData))
	for ii, jj := range clusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, err
}
