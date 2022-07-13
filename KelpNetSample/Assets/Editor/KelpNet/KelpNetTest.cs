using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using KelpNet;
using KelpNet.CPU;
using Real = System.Single;


public class KelpNetTest
{
    [MenuItem("KelpNet/Test/FNN")]
    public static void TestFNN()
    {
        Debug.Log("OK");
        const int epochs = 100000;
        float progress = 0;
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

        adam.SetUp(nn);
        Debug.Log("Training Start");
        
        for(int i = 0;i<epochs;i++)
        {
            Trainer.Train(nn,trainData[0],trainLabel[0],new MeanSquaredError<float>());
            Trainer.Train(nn,trainData[1],trainLabel[1],new MeanSquaredError<float>());
            Trainer.Train(nn,trainData[2],trainLabel[2],new MeanSquaredError<float>());
            Trainer.Train(nn,trainData[3],trainLabel[3],new MeanSquaredError<float>());

            adam.Update();
            progress = (float) i / epochs;

            EditorUtility.DisplayProgressBar("学習中", "epoch:" + i + "(" + (progress * 100).ToString("F0") + "%)",progress);
        }
        EditorUtility.ClearProgressBar();
        Debug.Log("Train Finished");
        Debug.Log("Test Start");
        foreach(float[] input in trainData)
        {
            NdArray<float> result = nn.Predict(input)[0];
            Debug.Log("input = " + input + " output = " + result);
        }
        Debug.Log("Test Finished");
    }

    [MenuItem("KelpNet/Test/XOR")]
    public static void TestXOR()
    {
        const int epochs = 10000;
        float progress = 0;
        Real[][] trainData =
        {
            new Real[] {0,0},
            new Real[] {0,1},
            new Real[] {1,0},
            new Real[] {1,1}
        };
        int[][] trainLabel =
        {
            new[] {0},
            new[] {1},
            new[] {1},
            new[] {0}
        };
        FunctionStack<Real> nn = new FunctionStack<Real>(
            new Linear<Real>(2,2,name:"l1 Linear"),
            new Sigmoid<Real>(name:"l1 Sigmoid"),
            new Linear<Real>(2,2,name:"l2 Linear")
        );

        Debug.Log("Training Start");
        for(int i = 0; i < epochs; i++)
        {
            for(int j = 0; j < trainData.Length; j++)
            {
                Trainer.Train(nn,new[]{trainData[j]},new[]{trainLabel[j]},new SoftmaxCrossEntropy<Real>(),new MomentumSGD<Real>());
            }
            progress = (float) i / epochs;

            EditorUtility.DisplayProgressBar("学習中", "epoch:" + i + "(" + (progress * 100).ToString("F0") + "%)",progress);
        }
        EditorUtility.ClearProgressBar();
        Debug.Log("Train Finished");
        Debug.Log("Test Start");
        foreach(Real[] input in trainData)
        {
            NdArray<Real> result = nn.Predict(input)[0];
            int resultIndex = Array.IndexOf(result.Data,result.Data.Max());
            Debug.Log(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
        }
        Debug.Log("Test Finished");

        ModelIO<Real>.Save(nn,"Assets/TrainedModel/samplexor.nn");
        Function<Real> testnn = ModelIO<Real>.Load("Assets/TrainedModel/samplexor.nn");

        Debug.Log("ReloadTest Start");
        foreach(Real[] input in trainData)
        {
            NdArray<Real> result = testnn.Predict(input)[0];
            int resultIndex = Array.IndexOf(result.Data,result.Data.Max());
            Debug.Log(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
        }
        Debug.Log("ReloadTest Finished");
        
    }
    
}
