package main

import (
	"PainTheMaster/mybraly/deeplearning"
	"PainTheMaster/mybraly/deeplearning/mnist"
	"encoding/gob"
	"fmt"
	"os"
)

type NNStore struct {
	W               [][][]float64
	DW              [][][]float64
	DiffW           [][][]float64
	ParamGradDecent struct {
		//Hyper parameters
		LearnRate float64
	}

	ParamMomentum struct {
		//Hyer parameters
		LearnRate  float64
		MomentRate float64
		//Working parameters
		moment [][][]float64
	}

	ParamAdaGrad struct {
		//Hyper parameters
		LearnRate float64
		//Working parameters
		Rep   int
		SqSum [][][]float64
	}

	ParamRMSProp struct {
		//Hyper parameters
		LearnRate float64
		DecayRate float64
		//Working parameters
		Rep     int
		ExpMvAv [][][]float64
	}

	ParamAdaDelta struct {
		//Hyper parameters
		DecayRate float64
		//WorkingParameters
		Rep         int
		ExpMvAvDW   [][][]float64
		ExpMvAvGrad [][][]float64
	}

	ParamAdam struct {
		//Hyper parameters
		LearnRate  float64
		DecayRate1 float64
		DecayRate2 float64
		//Working parameters
		Rep        int
		ExpMvAvPri [][][]float64
		ExpMvAvSec [][][]float64
	}
}

func main() {
	actFuncHidden := []string{deeplearning.LabelIdentity, deeplearning.LabelReLU, deeplearning.LabelReLU, deeplearning.LabelReLU, deeplearning.LabelReLU, deeplearning.LabelReLU}
	actFuncOut := deeplearning.LabelSoftMax

	mnistDataLen := mnist.MnistCols * mnist.MnistRows
	nodes := []int{mnistDataLen, 128, 64, 64, 32, 32, 10}
	neuralNet := deeplearning.Make(nodes, actFuncHidden, actFuncOut)

	neuralNet.ParamRMSProp.DecayRate = 0.9
	neuralNet.ParamRMSProp.LearnRate = 0.001

	neuralNet.ParamAdam.DecayRate1 = 0.9
	neuralNet.ParamAdam.DecayRate2 = 0.999
	neuralNet.ParamAdam.LearnRate = 0.0002

	neuralNet.ParamAdaDelta.DecayRate = 0.999

	neuralNet.ParamMomentum.LearnRate = 0.005
	neuralNet.ParamMomentum.MomentRate = 0.9

	neuralNet.ParamGradDecent.LearnRate = 0.005

	home := os.Getenv("HOME")
	fmt.Println("Obtained home:", home)

	trainImg, fileErr := os.Open(home + "/deeplearning/train-images-idx3-ubyte")
	fmt.Println("test filepath", home+"/deeplearning/train-images-idx3-ubyte")
	if fileErr != nil {
		fmt.Println("Training image file open error:", fileErr)
	}
	trainLabel, fileErr := os.Open(home + "/deeplearning/train-labels-idx1-ubyte")
	if fileErr != nil {
		fmt.Println("Training label file open error:", fileErr)
	}

	testImg, fileErr := os.Open(home + "/deeplearning/t10k-images-idx3-ubyte")
	if fileErr != nil {
		fmt.Println("Test image file open error:", fileErr)
	}
	testLabel, fileErr := os.Open(home + "/deeplearning/t10k-labels-idx1-ubyte")
	if fileErr != nil {
		fmt.Println("Training label file open error:", fileErr)
	}

	file, err := os.Open("/home/painthemaster/go/src/PainTheMaster/test/6layers/learningData-40.bin")
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&neuralNet)
	if err != nil {
		fmt.Println(err)
	}

	for epoch := 41; epoch <= 100; epoch++ {
		miniBatchSize := 50
		repetition := 1200
		neuralNet.Train(trainImg, trainLabel, miniBatchSize, repetition, deeplearning.LabelAdam)

		//	trainImg.Seek(0, 0)
		//	trainLabel.Seek(0, 0)

		file, err := os.Create("./6layers/learningData-" + fmt.Sprintf("%d", epoch) + ".bin")
		if err != nil {
			fmt.Println(err)
		}
		var tempNN NNStore
		tempNN.W = neuralNet.W
		tempNN.DW = neuralNet.DW
		tempNN.DiffW = neuralNet.DiffW

		encoder := gob.NewEncoder(file)
		encoder.Encode(tempNN)
		file.Close()

		testSize := 10000
		accuracyPct := neuralNet.Test(testImg, testLabel, testSize)
		fmt.Printf("Epoch-%d Test accuracy with %d samples: %f%%\n", epoch, testSize, accuracyPct)
		//	testImg.Close()
		//	testLabel.Close()
	}

}
