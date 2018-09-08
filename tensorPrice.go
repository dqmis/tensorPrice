package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"golang.org/x/oauth2/google"
	vision "google.golang.org/api/vision/v1"
)

type Label struct {
	Label       string
	Probability float32
}

type Labels []Label

func (l Labels) Len() int           { return len(l) }
func (l Labels) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }
func (l Labels) Less(i, j int) bool { return l[i].Probability > l[j].Probability }

var (
	graphFile  = "./model/shops_graph.pb"
	labelsFile = "./model/shops_labels.txt"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s <path-to-image>\n", filepath.Base(os.Args[0]))
	}

	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	if err := run(args[0]); err != nil {
		fmt.Fprint(os.Stderr, "%s\n", err.Error())
		os.Exit(1)
	}
}

func run(file string) error {
	ctx := context.Background()

	client, err := google.DefaultClient(ctx, vision.CloudPlatformScope)
	if err != nil {
		return err
	}

	service, err := vision.New(client)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}

	convImg := base64.StdEncoding.EncodeToString(b)

	req := &vision.AnnotateImageRequest{
		Image: &vision.Image{
			Content: convImg,
		},
		Features: []*vision.Feature{
			{
				Type: "DOCUMENT_TEXT_DETECTION",
			},
		},
	}

	batch := &vision.BatchAnnotateImagesRequest{
		Requests: []*vision.AnnotateImageRequest{req},
	}

	res, err := service.Images.Annotate(batch).Do()
	if err != nil {
		return err
	}

	re := regexp.MustCompile("([0-9]*[,])?[0-9]+")

	if annotations := res.Responses[0].TextAnnotations; len(annotations) > 0 {
		text := annotations[0].Description

		//fmt.Printf("Found text: %s\n", text)
		var arr = re.FindAllString(text, -1)
		var max float64
		max = 0

		for _, i := range arr {
			if j, err := strconv.ParseFloat(strings.Replace(i, ",", ".", -1), 64); err == nil && strings.Contains(i, ",") && strings.Contains(i, "%") == false {
				if j > max {
					max = j
				}
			}
		}
		fmt.Printf("Galutine kaina: %.2f Eur.\n", max)
	}

	//fmt.Printf("Not found text: %s\n", file)

	modelGraph, labels, err := loadGraphAndLabels()
	if err != nil {
		log.Fatal("unable to load graph and labels: %s", err)
	}

	session, err := tf.NewSession(modelGraph, nil)
	if err != nil {
		log.Fatal("unable to start a session: %s", err)
	}
	defer session.Close()

	tensor, err := normalizedImg(b)
	if err != nil {
		log.Fatal("unable to normalize img: %s", err)
	}

	result, err := session.Run(map[tf.Output]*tf.Tensor{
		modelGraph.Operation("input").Output(0): tensor,
	},
		[]tf.Output{
			modelGraph.Operation("final_result").Output(0),
		}, nil)
	if err != nil {
		log.Fatalf("unable to unference: %s", err)
	}

	topLabels := getTopFiveLabels(labels, result[0].Value().([][]float32)[0])

	for _, l := range topLabels {
		fmt.Printf("label: %s, probablity: %.2f%%\n", l.Label, l.Probability*100)
	}
	return nil
}

func getTopFiveLabels(labels []string, probabil []float32) []Label {
	var results []Label
	for i, p := range probabil {
		if i >= len(labelsFile) {
			break
		}
		results = append(results, Label{
			Label:       labels[i],
			Probability: p,
		})
	}
	sort.Sort(Labels(results))
	return results
}

func normalizedImg(body []byte) (*tf.Tensor, error) {
	t, err := tf.NewTensor(string(body[:]))
	if err != nil {
		return nil, err
	}

	graph, input, output, err := getNormalizedGraph()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	normalized, err := session.Run(map[tf.Output]*tf.Tensor{
		input: t,
	},
		[]tf.Output{
			output,
		}, nil)
	if err != nil {
		return nil, err
	}

	return normalized[0], nil
}

func getNormalizedGraph() (*tf.Graph, tf.Output, tf.Output, error) {
	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	output := op.Sub(s,
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				op.Cast(s, decode, tf.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		op.Const(s.SubScope("mean"), float32(117)))
	graph, err := s.Finalize()

	return graph, input, output, err
}

func loadGraphAndLabels() (*tf.Graph, []string, error) {
	model, err := ioutil.ReadFile(graphFile)
	if err != nil {
		return nil, nil, err
	}

	g := tf.NewGraph()
	if err = g.Import(model, ""); err != nil {
		return nil, nil, err
	}

	f, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var labels []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return g, labels, nil
}
