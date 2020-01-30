
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "read_cfg.h"

using namespace cv;
using namespace dnn;
using namespace std;

const char* keys =
"{help    h usage ? | | Usage examples: \n\t\t./detect_img --image=dog.jpg --cfg=net.data \n\t\t./detect_img --dir=/dirpath --cfg=net.data }"
"{image   i         |<none>| input image   }"
"{dir     d         |<none>| image dir     }"
"{cfg     c         |<none>| cfg file path }"
"{nms     n         |0.4| nms threshold    }"
"{thresh  t         |0.5| conf threshold   }"
;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out, float confThreshold, float nmsThreshold, vector<string>& classes);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
	if (!parser.has("cfg") || (!parser.has("image") && !parser.has("dir")))
	{
		parser.printMessage();
		return 0;
	}
	
	// Initialize the parameters
	float confThreshold = parser.get<float>("thresh"); // Confidence threshold
	float nmsThreshold = parser.get<float>("nms");  // Non-maximum suppression threshold

	string cfgFile = parser.get<String>("cfg");
	map<string, string> config;	
	ReadConfig(cfgFile, config);
	
	int inpWidth = atoi(FindInConfig(config, "width", "416").c_str());  // Width of network's input image
	int inpHeight = atoi(FindInConfig(config, "height", "416").c_str()); // Height of network's input image
    // Give the configuration and weight files for the model
    string modelConfiguration = FindInConfig(config, "cfg", "yolov3.cfg");
    string modelWeights = FindInConfig(config, "weights", "yolov3.weights");
    string classesFile = FindInConfig(config, "names", "coco.names");
	
    // Load names of classes
	vector<string> classes;
    ifstream ifs(classesFile);
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    // get image files list
	vector<String> files;    
    try {        
        if (parser.has("image"))
        {
            // Open the image file
            string str = parser.get<String>("image");
            ifstream ifile(str);
            if (ifile) {
				files.push_back(str);            	
            }
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            //outputFile = str;
        }
        else if (parser.has("dir"))
        {
            // Open the video file
            string pattern = parser.get<String>("dir") + "/*.jpg";
		    glob(pattern, files, false);
        }
		if (files.size() < 1) throw("error");
    }
    catch(...) {
        cout << "Could not open the input image/dir " << endl;
        return 0;
    }

    Mat frame, blob;	
    
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    for (size_t i=0; i<files.size(); i++) 
	{
        // get frame from the video
        frame = imread(files[i]);
        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Remove the bounding boxes with low confidence
        postprocess(frame, outs, confThreshold, nmsThreshold, classes);
        
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        //Mat detectedFrame;
        //frame.convertTo(detectedFrame, CV_8U);
        //if (parser.has("image")) imwrite(outputFile, detectedFrame);
        //else video.write(detectedFrame);
        
        imshow(kWinName, frame);
        waitKey();
    }

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, float confThreshold, float nmsThreshold, vector<string>& classes)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, classes);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
		{
			names[i] = layersNames[outLayers[i] - 1];
		}
    }
    return names;
}
