import csv
import optparse
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y, plot=False):
    
  # purity, completeness, thresholds
  precision, recall, pr_thresholds = precision_recall_curve(y_true, y, 1)
  fpr, tpr, thresholds = roc_curve(y_true, y)
    
  if plot:
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(fpr, 1-tpr)
    ax1.set_ylabel("false positive rate")
    ax1.set_xlabel("missed detection rate")

    ax2 = fig.add_subplot(122)
    ax2.plot(recall, precision)
    ax2.set_ylabel("purity")
    ax2.set_xlabel("completeness")

    plt.show()

  return precision, recall, pr_thresholds, fpr, tpr, thresholds

def precision_recall_curve(y_true, y, pos_label, step=0.04):

  pos_indices = np.where(y_true == pos_label)
  neg_indices = np.where(y_true != pos_label)

  thresholds = np.arange(0,1,step)
    
  precision = np.zeros(thresholds.shape)
  recall    = np.zeros(thresholds.shape)

  for i,threshold in enumerate(thresholds):
  #print np.where(y[pos_indices] >  threshold)[0].shape
    tp = float(np.where(y[pos_indices] >  threshold)[0].shape[0])
    fp = float(np.where(y[neg_indices] >  threshold)[0].shape[0])
    fn = float(np.where(y[pos_indices] <= threshold)[0].shape[0])
    #print tp, fp, fn
    try:
      precision[i] += tp / (tp + fp)
      recall[i]    += tp / (tp + fn)
    except ZeroDivisionError:
      print(threshold)
      #precision = np.concatenate((precision,np.array([1])))
   #recall = np.concatenate((recall,np.array([0])))

    precision = np.concatenate((precision,np.array([1])))
    recall = np.concatenate((recall,np.array([0])))
  return precision, recall, thresholds

def roc_curve(y_true, y, step=0.01):

  pos_indices = np.where(y_true == 1)
  neg_indices = np.where(y_true == 0)

  thresholds = np.arange(0,1,step)
    
  fpr = np.zeros(thresholds.shape)
  tpr = np.zeros(thresholds.shape)
  
  for i,threshold in enumerate(thresholds):
    try:
      fpr[i] += float(np.where(y[neg_indices] >= threshold)[0].shape[0]) / neg_indices[0].shape[0]
    except ZeroDivisionError:
      fpr[i] += 1
    try:
      tpr[i] += float(np.where(y[pos_indices] >= threshold)[0].shape[0]) / pos_indices[0].shape[0]
    except ZeroDivisionError:
      tpr[i] += 0
    
    fpr = np.concatenate((fpr,np.array([0])))
    tpr = np.concatenate((tpr,np.array([0])))

  return fpr, tpr, thresholds
    
def plotROC(y_true, labels, *args):

  colours = ["#3F88C5", "#4C4C9D", "#AF2BBF", "#7CEA9C", "#FFBA08"]

  fig = plt.figure()
  lw = 3
  ax1 = fig.add_subplot(121)
  
  ax1.set_ylabel("false positive rate")
  ax1.set_xlabel("missed detection rate")

  ax2 = fig.add_subplot(122)
  ax2.set_ylabel("purity")
  ax2.set_xlabel("completeness")

  for i,y in enumerate(args):

    precision, recall, _, fpr, tpr, thresholds = plot_roc_curve(y_true, y)
    print(labels[i])
    f = []
    m = []
    for t in [0.01, 0.05, 0.1]:
      print(t)
    #print thresholds
    
    #print np.where(fpr<=t)[0][0]
    #print "decision boundary: ", thresholds[np.where(fpr<=t)[0][0]]
    #print fpr[np.where(fpr<=t)[0]][0]
      print("mdr: ", 1-tpr[np.where(fpr<=t)[0]][0])
      m.append(1-tpr[np.where(fpr<=t)[0]][0])
    #print np.where(1-tpr<=t)
      print("fpr: ", fpr[np.where(1-tpr<=t)[0]][-1])
      f.append(fpr[np.where(1-tpr<=t)[0]][-1])
    if labels[i] == "combined":
      ax1.plot(1-tpr, fpr, color="k", lw=lw,zorder=100)
      ax2.plot(recall, precision, color="k", lw=lw,zorder=100)
      ax1.plot(1-tpr, fpr, label=labels[i], color=colours[i], lw=lw,zorder=100)
      ax2.plot(recall, precision, color=colours[i], lw=lw,zorder=100)
    else:
      ax1.plot(1-tpr, fpr, label=labels[i], color=colours[i], lw=lw)
      ax2.plot(recall, precision, color=colours[i], lw=lw)
  
  ax2.set_xlim(0,1)
  ax2.set_ylim(0,1)
  ax1.set_aspect(1./ax1.get_data_ratio())
  ax2.set_aspect(1./ax2.get_data_ratio())
  ax1.legend()
  plt.show()

  #print("swap_combined_fpr_r = [%lf, %lf, %lf]" % (f[0], f[1], f[2]))
  #print("swap_combined_mdr_r = [%lf, %lf, %lf]" % (m[0], m[1], m[2]))

def main():
  parser = optparse.OptionParser("[!] usage: python plotROC.py\n"+\
                                 "           -f <prediction file>")

  parser.add_option("-f", dest="predictionFile", type="string", \
    help="specify prediction file (csv cols: filename,label,prediction)")

  (options, args) = parser.parse_args()

  predictionFile = options.predictionFile

  files  = []
  y_true = []
  preds  = []

  with open(predictionFile, "r") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
      files.append(row[0].strip())
      y_true.append(int(row[1]))
      preds.append(float(row[2]))

  y_true = np.array(y_true)
  preds  = np.array(preds)

  print(y_true)
  print(preds)

  plotROC(y_true, [predictionFile], preds)

if __name__ == "__main__":
  main()
