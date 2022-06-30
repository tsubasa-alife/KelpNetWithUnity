using System;
using System.Collections;
using System.Collections.Generic;
using KelpNet;
using KelpNet.CPU;
using UnityEngine;

public class SampleSimpleFNN : MonoBehaviour
{
    const int epochs = 10000;
    FunctionStack<float> nn = new FunctionStack<float>(
        new Linear<float>(2,3,name:"l1 linear"),
        new Tanh<float>(name:"l1 tanh"),
        new Linear<float>(3,2,name:"l2 linear")
    );
    Adam<float> adam = new Adam<float>();
    
    float[][] trainData =
    {
        new float[] {1.0f,0.0f},
        new float[] {0.0f,1.0f},
        new float[] {-1.0f,0.0f},
        new float[] {0.0f,-1.0f}
    };
    float[][] trainLabel =
    {
        new float[] {0.0f,1.0f},
        new float[] {-1.0f,0.0f},
        new float[] {0.0f,-1.0f},
        new float[] {1.0f,0.0f}
    };
    // Start is called before the first frame update
    void Start()
    {
        adam.SetUp(nn);
        Debug.Log("Training Start");
        for(int i = 0;i<epochs;i++)
        {
            Trainer.Train(nn,trainData[0],trainLabel[0],new MeanSquaredError<float>());
            Trainer.Train(nn,trainData[1],trainLabel[1],new MeanSquaredError<float>());
            Trainer.Train(nn,trainData[2],trainLabel[2],new MeanSquaredError<float>());
            Trainer.Train(nn,trainData[3],trainLabel[3],new MeanSquaredError<float>());

            adam.Update();
        }
        Debug.Log("Train Finished");
        Debug.Log("Test Start");
        foreach(float[] input in trainData)
        {
            NdArray<float> result = nn.Predict(input)[0];
            Debug.Log("input = " + input + " output = " + result);
        }
        Debug.Log("Test Finished");
    }
}
