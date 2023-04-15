

|<p>ID1: 208560086</p><p>ID2: 301613501</p>|<p>**Deep Learning** </p><p>**2022-2023**</p>|![](Images/Aspose.Words.e53994a0-9a1f-4f49-98f5-4e48e764a315.001.png)|
| :- | :-: | -: |


**EX1 - DL Basics**
#
1. **General**

This readme file describes how to run the code in the notebook of exercise 1.

The code is written such that we train all the cases and show the required graphs on a *Tensorboard*.

1. **Run the code**
   Prior to the training or loading weight step, please run the following steps \ functions \ cells:
   1. Cell 1: Mount to google drive.
   1. Cell 2: Import libraries.
   1. Cell 3: Set the device that PyTorch will use for computation based on the available hardware resources.
   1. Cell 4: Load Tensorboard extension.
   1. Cell 5: Transform the data into a PyTorch tensor.
   1. Cell 6: Initializes the train\_dataset and test\_dataset, using the Fashion-MNIST dataset.
   1. Cell 7: Implementation of LeNet-5 and get\_optimizer method
   1. Cell 8: Run the train\_one\_ephoc function
   1. Cell 9: Run the evaluate the function
   1. Cell10: Run the get\_model function
   1. Cell 11: Run the train\_lenet5 function
   1. Cell 12: Run the test\_lenet5 function

1. **How to train**
   Run the *train\_lenet5* function as shown in the following table.

|**Regularization**|**Command**|
| :-: | :-: |
|Without Regularization|train\_lenet5(batch\_size=64, initial\_lr=0.01, num\_epochs=30, batch\_norm\_enabled=False, dropout\_enabled=False, weight\_decay\_enabled=False, weight\_decay=0.0001)|
|Batch Normalization|train\_lenet5(batch\_size=64, initial\_lr=0.01, num\_epochs=30, batch\_norm\_enabled=True, dropout\_enabled=False, weight\_decay\_enabled=False,weight\_decay=0.0001)|
|Dropout|train\_lenet5(batch\_size=64, initial\_lr=0.01, num\_epochs=30, batch\_norm\_enabled=False, dropout\_enabled=True, weight\_decay\_enabled=False,weight\_decay=0.0001)|
|Weight Decay|train\_lenet5(batch\_size=64, initial\_lr=0.01, num\_epochs=30, batch\_norm\_enabled=False, dropout\_enabled=False, weight\_decay\_enabled=True,weight\_decay=0.0001)|

Function Description:

Trains a LeNet-5 model on a given dataset and returns the final training and testing accuracies.
Args:
- batch\_size (int): the number of samples per batch to load in the data loaders.
- initial\_lr (float): the initial learning rate to use for the optimizer.
- num\_epochs (int): the number of epochs to train the model for.
- batch\_norm\_enabled (bool): whether to enable batch normalization in the model.
- dropout\_enabled (bool): whether to enable dropout in the model.
- weight\_decay\_enabled (bool): whether to enable weight decay in the model.
- weight\_decay (float): the weight decay value to use for the optimizer.
- weights\_path (str): (optional) the file path to save the trained model weights to. If no value is provided then wights\_path will be assigned with ‘lenet\_5\_bn{}\_dp{}\_wd{}\_weights.pth’ when {} will be true or false according to the function values.

Returns:
- train\_acc (float): the final training accuracy of the model on the train set.
- test\_acc (float): the final testing accuracy of the model on the test set.

1. <a name="_heading=h.gjdgxs"></a>**How to test with saved weights** 
   Run the *test\_lenet5* function as shown in the following table.

|**Regularization**|**Command**|
| :-: | :-: |
|Without Regularization|test\_lenet5(batch\_size=64, batch\_norm\_enabled=False, dropout\_enabled=False, weight\_decay\_enabled=False,weight\_decay=0.0001,weights\_path ='lenet5\_bnFalse\_dpFalse\_wdFalse\_weights.pth')|
|Batch Normalization|test\_lenet5(batch\_size=64, batch\_norm\_enabled=False, dropout\_enabled=False, weight\_decay\_enabled=False,weight\_decay=0.0001,weights\_path ='lenet5\_bnTrue\_dpFalse\_wdFalse\_weights.pth')|
|Dropout|test\_lenet5(batch\_size=64, batch\_norm\_enabled=False, dropout\_enabled=False, weight\_decay\_enabled=False,weight\_decay=0.0001,weights\_path ='lenet5\_bnFalse\_dpFalse\_wdTrue\_weights.pth')|
|Weight Decay|test\_lenet5(batch\_size=64, batch\_norm\_enabled=False, dropout\_enabled=False, weight\_decay\_enabled=False,weight\_decay=0.0001,weights\_path ='lenet5\_bnFalse\_dpFalse\_wdTrue\_weights.pth')|

Function Description:

Test a LeNet-5 model on a given dataset and return the final training and testing accuracies on a pre-trained model.
Args:
- batch\_size (int): the number of samples per batch to load in the data loaders.
- batch\_norm\_enabled (bool): whether to enable batch normalization in the model.
- dropout\_enabled (bool): whether to enable dropout in the model.
- weight\_decay\_enabled (bool): whether to enable weight decay in the model.
- weight\_decay (float): the weight decay value to use for the optimizer.
- weights\_path (str):  the file path to load the trained model weights to.

Returns:
- train\_acc (float): the final testing accuracy of the model on the train set.
- test\_acc (float): the final testing accuracy of the model on the test set.

1. By running Cell 20 you can see a **table of the summary
	
   `		`![](Images/Aspose.Words.e53994a0-9a1f-4f49-98f5-4e48e764a315.002.png)**
1. <a name="_heading=h.30j0zll"></a>By running Cell 21 (%tensorboard --logdir logs) you can see the Tensorboard visualization of all the graphs. Then follow the illustration below in order to choose the relevant graph\s you want to see. It is possible to compare all the train-test combinations.
1. <a name="_heading=h.s33bpoe7jdbk"></a>For example - if we want to compare the test and the train of batch normalization we will choose test/lenet\_5\_**bnTrue**… and train/lenet5\_**bnTrue**.

   `	`![](Images/Aspose.Words.e53994a0-9a1f-4f49-98f5-4e48e764a315.003.png)
