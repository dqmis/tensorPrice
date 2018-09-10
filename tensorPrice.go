package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"os"
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

type Response struct {
	Shop  string
	Price float64
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
	http.HandleFunc("/", makeResponse)
	http.ListenAndServe(":8080", nil)
}

func enableCors(w *http.ResponseWriter) {
	(*w).Header().Set("Access-Control-Allow-Origin", "*")
}

func makeResponse(w http.ResponseWriter, r *http.Request) {
	enableCors(&w)
	resp, header, err := r.FormFile("file")
	_ = header
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Close()

	byteImg, err := ioutil.ReadAll(resp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	retailer, err := runImg(string(byteImg))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	price, err := runText(base64.StdEncoding.EncodeToString(byteImg))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	sendResp := Response{retailer.Label, price}

	js, err := json.Marshal(sendResp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(js)
}

func runImg(img string) (Label, error) {
	empty := Label{}
	modelGraph, labels, err := loadGraphAndLabels()
	if err != nil {
		return empty, err
	}

	session, err := tf.NewSession(modelGraph, nil)
	if err != nil {
		return empty, err
	}
	defer session.Close()

	tensor, err := normalizedImg(img)
	if err != nil {
		return empty, err
	}

	result, err := session.Run(map[tf.Output]*tf.Tensor{
		modelGraph.Operation("input").Output(0): tensor,
	},
		[]tf.Output{
			modelGraph.Operation("final_result").Output(0),
		}, nil)
	if err != nil {
		return empty, err
	}

	return getLabels(labels, result[0].Value().([][]float32)[0]), nil
}

func runText(img string) (float64, error) {
	ctx := context.Background()

	client, err := google.DefaultClient(ctx, vision.CloudPlatformScope)
	if err != nil {
		log.Fatal(err)
	}

	service, err := vision.New(client)
	if err != nil {
		return 0, err
	}

	req := &vision.AnnotateImageRequest{
		Image: &vision.Image{
			Content: img,
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
		return 0, err
	}

	re := regexp.MustCompile("([0-9]*[,])?[0-9]+")

	if annotations := res.Responses[0].TextAnnotations; len(annotations) > 0 {
		text := annotations[0].Description

		var arr = re.FindAllString(text, -1)
		var max float64
		max = 0

		for _, i := range arr {
			if j, err := strconv.ParseFloat(strings.Replace(i, ",", ".", -1), 64); err == nil && strings.Contains(i, ",") && strings.Contains(i, "%") == false {
				if j > max && j != 21 {
					max = j
				}
			}
		}
		return max, nil
	}
	return 0, nil
}

func getLabels(labels []string, probabil []float32) Label {
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
	return results[0]
}

func normalizedImg(img string) (*tf.Tensor, error) {
	t, err := tf.NewTensor(img)
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
