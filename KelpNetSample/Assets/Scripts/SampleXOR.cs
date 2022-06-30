using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using KelpNet;
using KelpNet.CPU;
using Real = System.Single;

public class SampleXOR : MonoBehaviour
{
    const int epochs = 10000;
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
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Training Start");
        for(int i = 0; i < epochs; i++)
        {
            for(int j = 0; j < trainData.Length; j++)
            {
                Trainer.Train(nn,new[]{trainData[j]},new[]{trainLabel[j]},new SoftmaxCrossEntropy<Real>(),new MomentumSGD<Real>());
            }
        }
        Debug.Log("Test Start");
        foreach(Real[] input in trainData)
        {
            NdArray<Real> result = nn.Predict(input)[0];
            int resultIndex = Array.IndexOf(result.Data,result.Data.Max());
            Debug.Log(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
        }
        Debug.Log("Test Finished");

        ModelIO<Real>.Save(nn,"test.nn");
        Function<Real> testnn = ModelIO<Real>.Load("test.nn");

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
