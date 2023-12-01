
from sklearn import metrics
from sklearn.cluster import KMeans

from collections import Counter
from scipy.optimize import linear_sum_assignment

from sklearn import metrics

# put your image generator here
imgs_new = []

plt.figure(figsize=(6,6))

# for loop for generating 9 images
for n in range(9):

    # getting random input using inbuild random function of torch
    inp_rnd = torch.rand(1, d).to(device)

    # gnerating images using decoder given in the script
    with torch.no_grad():
        imgs_gen = decoder(inp_rnd)

    # plotting the images in 3x3 matrix as mentioned 
    plt.subplot(3, 3, len(imgs_new) + 1)
    plt.axis('off')
    plt.imshow(imgs_gen.cpu().squeeze().numpy(), cmap='gist_gray')
    imgs_new.append(imgs_gen)

# tighting the layout to get bigger images
plt.tight_layout()
plt.show()


# put your clustering accuracy calculation here
def enc_opts(enc, dataLoader, dev):

    # using encoder output as the input
    enc.eval()
    enc_opts = []
    t_labs = []

    with torch.no_grad():

        for img, lab in dataLoader:

            img = img.to(dev)
            t_labs.extend(lab.cpu().numpy())

            enc_data = enc(img)
            enc_opts.extend(enc_data.cpu().numpy())

    return np.array(enc_opts), np.array(t_labs)


def cls_kmeans(enc_opts, no_cls):
    
    # applying k means on the encoder outputs of the given dataset 48000 images and 10 clusters
    kmen = KMeans(n_clusters=no_cls, random_state=0)
    cls_assig = kmen.fit_predict(enc_opts)
    return cls_assig


def cal_acc(t_labs, cls_assig):

    # confusion matrix to check which true label needs to assign to which  predict label for the kmean algo
    conf_mat = metrics.confusion_matrix(t_labs, cls_assig)

    # getting the indexes using Hungarian algorithm
    r_idx, c_idx = linear_sum_assignment(-conf_mat)

    # calulating accuracy
    acc = conf_mat[r_idx, c_idx].sum() / len(t_labs)

    return acc

enc_opts, t_labs = enc_opts(encoder, train_loader, device)

no_cls = 10
cls_assig = cls_kmeans(enc_opts, no_cls)

acc = cal_acc(t_labs, cls_assig)

print(f"Accuracy obtained using Hungarian algorithm: {acc * 100:.2f}%")

