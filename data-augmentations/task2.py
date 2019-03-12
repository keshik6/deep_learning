# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:38:42 2019

@author: Keshik
"""

# In[0] Import required libraries
import torch
import numpy as np
import torchvision.models as models
from torchvision import transforms
from data_loader import get_data_loader, get_mean_and_std
from tqdm import tqdm

# In[3] Validation function 

def test(model, device, test_loader):
    model.eval()
    
    running_loss = 0
    running_correct = 0
    
    clf_matrix = torch.zeros(1000, 1000)
    
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            #print(data.size(), target.size())
            target = target.long()
            data, target = data.to(device), target.to(device)
            bs, ncrops, c, h, w = data.size()

            output = model(data.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1)

            running_loss += criterion(output, target) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            running_correct += pred.eq(target.view_as(pred)).sum().item()
            
            for t, p in zip(target.view(-1), pred.view(-1)):
                clf_matrix[t.long(), p.long()] += 1
                
            del data, target, output
            torch.cuda.empty_cache()
            #print("running_loss = {}, running_correct = {}".format(running_loss, running_correct))
            
                
    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item()/num_samples

    print('test_loss: {:.4f}, test_accuracy: {}/{} ({:.3f})'.format(
        avg_test_loss, running_correct, num_samples,
        running_correct / num_samples))
    
    clf_report = clf_matrix.diag()/clf_matrix.sum(1)
    
    return avg_test_loss, running_correct/num_samples, clf_report


# In[4] Main function here

def task2(images_dir, labels_file, num_images = 250, batch_size = 5, five_crop = True):
    
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Available device = ", device)
    
    crop_transform = None
    
    if five_crop == True:
        crop_transform = transforms.FiveCrop(224)
        print("Testing with Five Crop Transformation")
    else:
        crop_transform = transforms.TenCrop(224)
        print("Testing with Ten Crop Transformation")
    
    model = models.resnet50(pretrained=True).to(device)
    
    transformations= transforms.Compose([transforms.Resize(280), 
                                          crop_transform, 
                                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                          ])
    
    # Test without pixel normalization
    print("Testing of images without pixel normalization")
    val_loader = get_data_loader(images_dir, num_images, labels_file, transformations, batch_size)
    loss_1, accuracy_1, clf_report_1 = test(model, device, val_loader)
    torch.cuda.empty_cache()
    
    # With normalization
    mean = [0.4813360025834064, 0.45517925782495616, 0.4050549896240234] 
    std = [0.27411816110629167, 0.26685855695506344, 0.2806123479614292]
    
    transformations_normalize = transforms.Compose([transforms.Resize(280), 
                                          crop_transform, 
                                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                          transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                          ])

    # Test with pixel normalization
    print("\nTesting of images with pixel normalization")
    val_loader = get_data_loader(images_dir, num_images, labels_file, transformations_normalize, batch_size)
    loss_2, accuracy_2, clf_report_2 = test(model, device, val_loader)
    torch.cuda.empty_cache()
    
    return loss_2, accuracy_2
     
#task2(True)
#task2(False)
# In[6] Test the functions here

# Initialize object
#transformations = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
#dataset = ImageNetValDataSet("./imagenet_data/imagespart/", "./imagenet_data/synset_words.txt",
#                             transforms=transformations)
#images = []
#titles = []
#img_list = np.random.randint(1000, size=(10))
#
#for i in img_list:
#    image, label = dataset.__getitem__(i)
#    images.append(image)
#    titles.append(label)
#
#plot_image(images, titles)



