package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
)

const (
	// SiteNum is the number of sites.
	SiteNum int = 100
	// InstNum is the number of instances.
	InstNum int = 90
	// TestNum is the number of instances of testing.
	TestNum int = 90
	// OpenTestNum is the number of instances for open-world testing.
	OpenTestNum int = 9000

	// FolderWeight is the folder for weight learning.
	FolderWeight = "batch/"
	// FolderOpen is the folder for the open-world.
	FolderOpen = "batch/"
	// FolderTrain is the folder for training.
	FolderTrain = "batch/"
	// FolderTest is the folder for testing.
	FolderTest = "batch/"
	// FeatureSuffix is the suffix of files containing features
	FeatureSuffix = "s"

	// FeatNum is the number of extracted features to consider.
	FeatNum int = 1225
	// NeighbourNum is the number of neighbours in kNN.
	NeighbourNum int = 2
	// KReco is the number of neighbours for distance learning.
	KReco int = 5
	// p is the value used for L^p-norm distance metric. p=1 (taxicab) norm is
	// most efficient and Wang found increasing p>1 to have insignificant effect
	// on classifier accuracy.
	P int = 1
	// Rounds
	Rounds = 800
)

// Struct for holding our distances computation results between two points
// Re-consider naming PointPairNorm
type NormDataPoint struct {
	Idx       int
	LPNorm    float64
	FeatNorms [FeatNum]float64
}

// Struct for sorting by LP-norm distance
type ByLPNorm []NormDataPoint

func (a ByLPNorm) Len() int           { return len(a) }
func (a ByLPNorm) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByLPNorm) Less(i, j int) bool { return a[i].LPNorm < a[j].LPNorm }

func SortByLPNorm(a []NormDataPoint) {
	sort.Sort(ByLPNorm(a))
}

// Struct for sorting distances by a particular feature
type ByFeatNorm struct {
	NormData []NormDataPoint
	Idx      int
}

func (a ByFeatNorm) Len() int      { return len(a.NormData) }
func (a ByFeatNorm) Swap(i, j int) { a.NormData[i], a.NormData[j] = a.NormData[j], a.NormData[i] }
func (a ByFeatNorm) Less(i, j int) bool {
	return a.NormData[i].FeatNorms[a.Idx] < a.NormData[j].FeatNorms[a.Idx]
}

func SortByFeatNorm(a []NormDataPoint, i int) {
	sort.Sort(ByFeatNorm{a, i})
}

// Struct for holding the associated weight of each feature
type FeatWeight struct {
	Idx int
	Weight float64
}

// Type for sorting by weight
type ByWeight [FeatNum]FeatWeight

func (a ByWeight) Len() int { return len(a) }
func (a ByWeight) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByWeight) Less(i, j int) bool { return a[i].Weight < a[j].Weight }

func SortByWeight(a []FeatWeight) {
	sort.Sort(ByWeight(a))
}

// Initialize our weights with random floats in range (0.5, 1.5)
func initWeight(w [Featnum]FeatWeight) {
	for i := range w {
		w[i].Idx = i
		w[i].Weight = rand.Float64() + 0.5
	}
}

func 

// Compute all Norms between a given point
// Fix args here
func lPNorm([]NormDataPoint) (d float64) {
	for i := 0; i < FeatNum; i++ {
		if f1[i] != -1 && f2[i] != -1 {
			d += weight[i] * math.Abs(f1[i]-f2[i])
		}
	}
	if p > 1 {
		d = math.Pow(d, 1.0/p)
	}
	return
}

// maybe replace start, end w/ what we're actually talking about
func determineWeights(feat [][]float64, weight []float64, start, end int) {
	distList := make([]IndexedFloat64, SiteNum*InstNum)
	recoGoodList := make([]int, KReco)
	recoBadList := make([]IndexedWeight, KReco)
	log.Printf("starting to learn distance...")
	for i := start; i < end; i++ {
		fmt.Printf("\r\tdistance... %d (%d-%d)", i, start, end)

		curSite := int(i / InstNum)

		// calculate the distance to every other instance
		for j := 0; j < SiteNum*InstNum; j++ {
			distList[j] = j, dist(feat[i], feat[j], weight)
		}

		// Find k_reco points closest to self of same site (S_good)
		sliceGood = sort(distList[curSite*InstNum : (curSite+1)*InstNum])[1 : KReco+1]

		// Find k_reco points closest to self of other sites (S_bad)
		sliceBad = sort(distList[:curSite*InstNum-1] + distList[(curSite+1)*InstNum+1:])[1 : KReco+1]

		// Find number of releveant bad instances for each feature (n_bad)
		distMaxGood := make([]IndexedFloat64, FeatNum)
		for j := 0, j < FeatNum, j++ {
			distMaxGood[j] = 

			if IndexedDist.Float <= distMaxGood {
				nBad++
			}
		}
	}
}

func readFile(folder, name string, sites, instances int, openWorld bool) (feat [][]NormDataPoint) {
	// We read the results from running our feature extractor on our dataset into
	// the featureset datastructure that looks like this
	// [ [ [ features from instance ([]float64) ] [ ]  ... ] [ [ ] [ ] ... ] ... ]
	//   {                     s i t e                     }
	// {                            f e a t u r e s e t                          }

	featureset = make([]NormDataPoint, sites)
	for i := 0; i < len(featureset); i++ {
		featureset[i] = make([]NormDataPoint, instances)
	}

	// iterate over each site-instance
	for curSite := 0; curSite < sites; curSite++ {
		failCount := 0
		for curInst := 0; curInst < instances; curInst++ {
			// read the next file with features, continuing to the next instance if one is missing
			// this was the approach in the original code, so presumably we need it to parse their data
			var features string
			for {
				filename := path.Join(folder,
					strconv.Itoa(curSite)+"-"+strconv.Itoa(curInst+failCount)+FeatureSuffix)
				if openWorld {
					// only one instance in the open world
					filename = path.Join(folder, strconv.Itoa(curSite)+FeatureSuffix)
				}
				d, err := ioutil.ReadFile(filename)
				if err != nil {
					if failCount > 1000 {
						log.Fatalf("failed to find instance files to read (at least 1000 instances missing) for filename %s", filename)
					}
					failCount++
					continue
				}
				features = string(d)
				break
			}

			// extract features
			fCount := 0
			for _, f := range strings.Split(features, " ") {
				if f == "'X'" {
					feat[curSite*instances+curInst][fCount] = -1
					fCount++
				} else if f != "" {
					feat[curSite*instances+curInst][fCount] = parseFeatureString(f)
					fCount++
				}
			}
		}
	}

	log.Printf("loaded instances: %s", name)
	return
}

func parseFeatureString(c string) float64 {
	val, err := strconv.ParseFloat(c, 64)
	if err != nil {
		panic(err)
	}
	return val
}

func main() {
	// load features from collected traces for (in this order):
	// - weight learning (always closed world)
	// - training and testing (closed world)
	// - open world
	feat := readFile(FolderWeight, "main", SiteNum, InstNum, false)
	trainclosedfeat := readFile(FolderTrain, "training", SiteNum, TestNum, false)
	testclosedfeat := readFile(FolderTest, "testing", SiteNum, TestNum, false)
	openfeat := readFile(FolderOpen, "open", OpenTestNum, 1, true)

	// Make this an array instead of a slice?
	weight := make([]float64, FeatNum)
	initWeight(weight)
	for r := 0; r < Rounds; r++ {
		determineWeights(weight)
	}

	// calculate the accuracy in terms of true positives and true negatives
	tp, tn := accuracy(trainclosedfeat, testclosedfeat, openfeat, weight)
	log.Printf("Accuracy: %f %f", tp, tn)

	f, err := os.OpenFile("weights."+strconv.Itoa(int(time.Now().Unix())),
		os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	for i := 0; i < FeatNum; i++ {
		fmt.Fprintf(f, "%f ", weight[i]*1000)

}
