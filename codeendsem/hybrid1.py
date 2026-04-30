# HYBRID MODEL 1 -> GEORESNET + RANDOMFOREST
import os, joblib, torch, numpy as np, tensorflow as tf
import torch.nn as nn, torchvision.models as models
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
class GeoResNet(nn.Module):
    def __init__(self, in_channels=6):
        super(GeoResNet, self).__init__()
        try: weights = models.ResNet18_Weights.IMAGENET1K_V1; self.resnet = models.resnet18(weights=weights)
        except: self.resnet = models.resnet18(pretrained=True)

        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=original_conv1.out_channels,
                                      kernel_size=original_conv1.kernel_size, stride=original_conv1.stride,
                                      padding=original_conv1.padding, bias=False)
        with torch.no_grad():
            mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
            self.resnet.conv1.weight.copy_(mean_weights.repeat(1, in_channels, 1, 1))

        self.resnet.fc = nn.Identity()

    def forward(self, x): return self.resnet(x)

class HybridWideAndDeep:
    def __init__(self, in_channels=6):
        self.deep_model = GeoResNet(in_channels=in_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deep_model.to(self.device)
        self.deep_model.eval()

    def extract_embeddings(self, image_patches, batch_size=32):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(image_patches).float())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                embeddings.append(self.deep_model(batch[0].to(self.device)).cpu().numpy())
        return np.vstack(embeddings) if embeddings else np.zeros((0, 512), dtype=np.float32)

    def fit(self, X_patches, X_tabular, y):
        self.pretrain_cnn(X_patches, y)
        print(f"Extracting deep features from {len(X_patches)} patches...")
        deep_features = self.extract_embeddings(X_patches)
        hybrid_features = np.hstack([deep_features, X_tabular])

        self.classifier = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=-1)
        self.classifier.fit(hybrid_features, y)
        print(f"RF Training Accuracy: {accuracy_score(y, self.classifier.predict(hybrid_features)):.4f}")

    def predict_proba(self, X_patches, X_tabular):
        return self.classifier.predict_proba(np.hstack([self.extract_embeddings(X_patches), X_tabular]))[:, 1]

    def save(self, output_dir="/content/drive/MyDrive/Project_WISE/saved_model"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        torch.save(self.deep_model.state_dict(), os.path.join(output_dir, "geo_resnet.pth"))
        joblib.dump(self.classifier, os.path.join(output_dir, "rf_classifier.joblib"))
        print(f"Models saved to: {output_dir}")

    def pretrain_cnn(self, X_train, y_train, epochs=15):
        print(f"\n--- Fine-Tuning CNN for {epochs} epochs ---")
        self.deep_model.train()
        self.deep_model.resnet.fc = nn.Linear(512, 2).to(self.device)
        optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), batch_size=32, shuffle=True)

        for ep in range(epochs):
            total_loss, correct, total = 0, 0, 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.deep_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == batch_y).sum().item()
                total += batch_y.size(0)
            print(f"Epoch {ep+1:02d} | Loss: {total_loss/len(loader):.4f} | Acc: {100*correct/total:.2f}%")
        self.deep_model.resnet.fc = nn.Identity()
        self.deep_model.eval()

def load_dataset(tfrecord_path):
    print(f"Loading data from: {tfrecord_path}")
    X_list, y_list = [], []
    for raw_record in tqdm(tf.data.TFRecordDataset([tfrecord_path])):
        ex = tf.train.Example.FromString(raw_record.numpy())
        y_list.append(ex.features.feature['label'].int64_list.value[0])
        img_bytes = ex.features.feature['image'].bytes_list.value[0]
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(64, 64, 6).transpose(2, 0, 1)
        X_list.append(img)
    return np.array(X_list), np.array(y_list)

if __name__ == "__main__":
    tfrecord_file = "/content/drive/MyDrive/Project_WISE/data/merged_64x64_wells.tfrecord"
    X_patches, y = load_dataset(tfrecord_file)
    X_tabular = np.hstack([X_patches.mean(axis=(2, 3)), X_patches.std(axis=(2, 3))])

    X_p_train, X_p_test, X_t_train, X_t_test, y_train, y_test = train_test_split(X_patches, X_tabular, y, test_size=0.2, random_state=42, stratify=y)

    model = HybridWideAndDeep(in_channels=6)
    model.fit(X_p_train, X_t_train, y_train)

    print("\n--- Evaluation on Test Set ---")
    preds = (model.predict_proba(X_p_test, X_t_test) > 0.5).astype(int)
    print(classification_report(y_test, preds, target_names=['Not Suitable', 'Suitable']))
    model.save()