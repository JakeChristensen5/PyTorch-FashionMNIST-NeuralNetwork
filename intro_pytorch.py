import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Helper function to create and return the transform
def create_transform():
   """
   Creates and returns a composed transform for preprocessing the dataset images.
   """
   normalize_transform = transforms.Normalize((0.5,), (0.5,))  #Normalization parameters
   composed_transform = transforms.Compose([
       transforms.ToTensor(),  #Convert images to PyTorch tensors
       normalize_transform  #Normalize the images
   ])
   return composed_transform

#Helper function to load the dataset
def load_fashion_mnist_dataset(is_training=True, root_dir='./data', transform=None):
   """
   Loads the FashionMNIST dataset as specified by the training flag.
   """
   mnist_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=is_training,
                                                     download=True, transform=transform)
   return mnist_dataset

def get_data_loader(training=True):
   """
   Retrieves a DataLoader for the FashionMNIST dataset for either training or testing.


   INPUT:
       training: A boolean flag to indicate whether to load the training dataset (default True).
   RETURNS:
       DataLoader for the specified dataset.
   """
   preprocessing_transform = create_transform()
   mnist_dataset = load_fashion_mnist_dataset(is_training=training, transform=preprocessing_transform)
   mnist_data_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=training)  #Shuffle only if training is True
   return mnist_data_loader

def build_model():
   """
   TODO: implement this function.


   INPUT:
       None


   RETURNS:
       An untrained neural network model
   """
   model = nn.Sequential(
       nn.Flatten(),
       nn.Linear(28*28, 128),
       nn.ReLU(),
       nn.Linear(128, 64),
       nn.ReLU(),
       nn.Linear(64, 10)
   )
   return model

#Helper function to perform a single training step
def perform_training_step(model, data, target, optimizer, criterion):
   """
   Performs a single training step, including forward pass, loss calculation,
   backpropagation, and optimizer step.
   """
   optimizer.zero_grad()  #Zero the gradients
   output = model(data)  #Compute the model output
   loss = criterion(output, target)  #Calculate loss
   loss.backward()  #Backpropagate the loss
   optimizer.step()  #Update the model parameters
   return output, loss

#Helper function to update metrics after each batch
def update_metrics(output, target, loss, metrics):
   """
   Updates the running totals for loss, correct predictions, and total samples.
   """
   _, predicted = torch.max(output.data, 1)
   metrics['total_loss'] += loss.item() * target.size(0)
   metrics['correct'] += (predicted == target).sum().item()
   metrics['total'] += target.size(0)

def train_model(model, train_loader, criterion, T):
   optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   model.train()  #Set model to training mode


   for epoch in range(T):
       metrics = {'total_loss': 0, 'correct': 0, 'total': 0}


       for data, target in train_loader:
           output, loss = perform_training_step(model, data, target, optimizer, criterion)
           update_metrics(output, target, loss, metrics)


       avg_loss = metrics['total_loss'] / metrics['total']
       accuracy = 100 * metrics['correct'] / metrics['total']
       #Corrected line below with double quotes for the f-string
       print(f"Train Epoch: {epoch}   Accuracy: {metrics['correct']}/{metrics['total']}({accuracy:.2f}%) Loss: {avg_loss:.3f}")

#Helper function for a single evaluation step
def evaluate_batch(model, data, target, criterion):
   """
   Performs a forward pass and computes the loss for a single batch during evaluation.
   """
   output = model(data)  #Forward pass
   loss = criterion(output, target)  #Compute the loss
   return output, loss

#Helper function to print evaluation results
def print_evaluation_results(avg_loss, accuracy, show_loss):
   """
   Prints the average loss and accuracy of the model on the evaluation dataset.
   """
   if show_loss:
       print(f'Average Loss: {avg_loss:.4f}')
   print(f'Accuracy: {accuracy:.2f}%')

def evaluate_model(model, test_loader, criterion, show_loss=True):
   model.eval()  #Set the model to evaluation mode
   correct = 0
   total = 0
   total_loss = 0.0


   with torch.no_grad():  #Disable gradient computation
       for data, target in test_loader:
           output, loss = evaluate_batch(model, data, target, criterion)
           total_loss += loss.item() * data.size(0)
           _, predicted = torch.max(output.data, 1)
           total += target.size(0)
           correct += (predicted == target).sum().item()


   avg_loss = total_loss / total
   accuracy = 100 * correct / total


   print_evaluation_results(avg_loss, accuracy, show_loss)

#Given class names for mapping
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


#Helper function to print the top predictions
def print_top_predictions(probabilities, labels, top_k=3):
   """
   Prints the top K predictions given the probabilities and label indices.
   """
   for i in range(top_k):
       label_index = labels[0, i].item()  # Get the index for each label
       label_name = class_names[label_index]  # Map index to label name
       label_prob = probabilities[0, i].item() * 100  # Convert probability to percentage
       print(f"{label_name}: {label_prob:.2f}%")

def predict_label(model, test_images, index):
   model.eval()  #Set the model to evaluation mode
   with torch.no_grad():  #Disable gradient computation
       #Extract the image and add an extra batch dimension
       img = test_images[index].unsqueeze(0)
       logits = model(img)
       probabilities = F.softmax(logits, dim=1)
       top_probs, top_labels = torch.topk(probabilities, 3)  #Get the top 3 predictions
      
       #Utilize helper function to print top 3 predictions
       print_top_predictions(top_probs, top_labels)


# def comprehensive_test():
#    #Load data
#    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
#    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
  
#    #Assuming the dataset is the same for both training and testing for demonstration
#    train_loader = test_loader


#    #Build model
#    model = build_model()


#    #Define loss function
#    criterion = torch.nn.CrossEntropyLoss()


#    #Train model (For demonstration, you might train for a very small number of epochs)
#    train_model(model, train_loader, criterion, T=5)


#    #Evaluate model
#    evaluate_model(model, test_loader, criterion)


#    #Predict label for an image
#    test_images, _ = next(iter(test_loader))
#    predict_label(model, test_images, 0)  #Predict label for the first image


if __name__ == '__main__':
   #comprehensive_test()
   '''
   Feel free to write your own test code here to exaime the correctness of your functions.
   Note that this part will not be graded.
   '''


