package main

import (
	"PainTheMaster/mybraly/deeplearning"
	"PainTheMaster/mybraly/deeplearning/mnist"
	"encoding/gob"
	"fmt"
	"os"
)

func main() {
	actFuncHidden := []string{deeplearning.LabelIdentity, deeplearning.LabelReLU, deeplearning.LabelReLU, deeplearning.LabelReLU}
	actFuncOut := deeplearning.LabelSoftMax

	mnistDataLen := mnist.MnistCols * mnist.MnistRows
	nodes := []int{mnistDataLen, 32, 32, 32, 10}
	neuralNet := deeplearning.Make(nodes, actFuncHidden, actFuncOut)

	neuralNet.ParamRMSProp.DecayRate = 0.9
	neuralNet.ParamRMSProp.LearnRate = 0.001

	neuralNet.ParamAdam.DecayRate1 = 0.9
	neuralNet.ParamAdam.DecayRate2 = 0.999
	neuralNet.ParamAdam.LearnRate = 0.001

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

	miniBatchSize := 50
	repetition := 200
	errHist := neuralNet.Train(trainImg, trainLabel, miniBatchSize, repetition, deeplearning.LabelAdam)

	fmt.Println("Training finished:")
	for i := range errHist {
		fmt.Printf("%d,%f\n", i+1, errHist[i])
	}

	trainImg.Close()
	trainLabel.Close()

	testImg, fileErr := os.Open(home + "/deeplearning/t10k-images-idx3-ubyte")
	if fileErr != nil {
		fmt.Println("Test image file open error:", fileErr)
	}
	testLabel, fileErr := os.Open(home + "/deeplearning/t10k-labels-idx1-ubyte")
	if fileErr != nil {
		fmt.Println("Training label file open error:", fileErr)
	}

	testSize := 10000
	accuracyPct := neuralNet.Test(testImg, testLabel, testSize)
	fmt.Printf("Test accuracy with %d samples: %f%%\n", testSize, accuracyPct)

	file, err := os.Create("./learningData.bin")
	if err != nil {
		fmt.Println(err)
	}

	encoder := gob.NewEncoder(file)
	encoder.Encode(neuralNet)
	fmt.Println("learningData.bin stored")
	file.Close()

}
