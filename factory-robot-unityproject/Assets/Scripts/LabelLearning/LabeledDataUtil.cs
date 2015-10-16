using UnityEngine;
using System.Collections;

public static class LabeledDataUtil {

	/** Render camera, save snapshot and label */
	public static void Snapshot(Camera agentCamera,
	                            RenderTexture cameraRenderTexture,
	                            Texture2D readerTexture,
	                            string trainingFilePath,
	                            string captureFilePrefix,
	                            string labelFileName,
	                            string[] recordCategoryTags,
	                            int nDirectionSensors,
	                            string wallTag,
	                            int counter,
	                            float wallHitMultiplier) {

        agentCamera.Render();

        float[] sensorData = GetSensorInfo(agentCamera,
		                                   recordCategoryTags,
		                                   nDirectionSensors,
		                                   wallTag,
		                                   wallHitMultiplier);

        string labelVector = "";
        foreach (float a in sensorData) {
            labelVector += a.ToString();
            labelVector += ",";
        }
        //Debug.Log(labelVector);

        Color32[] currentImage = MiscUtils.getCurrentCameraImage(cameraRenderTexture, readerTexture);

        int downsampleFactor = 2;
        currentImage = MathUtils.downSampleImg(currentImage, downsampleFactor);

        Texture2D tex = new Texture2D(cameraRenderTexture.width / downsampleFactor, cameraRenderTexture.height / downsampleFactor);
        tex.SetPixels32(currentImage);

        FileUtils.SaveTextureToFile(tex, trainingFilePath + captureFilePrefix + counter.ToString("D6") + ".png");

        string trainingLine = labelVector + "\n";
        FileUtils.AppendStringToFile(trainingFilePath + labelFileName, trainingLine);
        
        
	}


	private static float[] GetSensorInfo(Camera agentCamera,
	                                     string[] recordCategoryTags,
	                                     int nDirectionSensors,
	                                     string wallTag,
	                                     float wallHitMultiplier) {
		int camWidth = agentCamera.pixelWidth;
		int nCategories = recordCategoryTags.Length;
		float[] signals = new float[nDirectionSensors*nCategories];
		
		float alpha = Mathf.Deg2Rad * agentCamera.transform.eulerAngles.x;
		float beta = 0.5f * Mathf.Deg2Rad * agentCamera.fieldOfView;
		int horizonY = Mathf.RoundToInt(agentCamera.pixelHeight * (0.5f + (Mathf.Tan(alpha) / (2.0f * Mathf.Tan(beta)))));
		
		int pixelsPerDirectionSensor = camWidth / nDirectionSensors;
		float pixelContribution = 1.0f / pixelsPerDirectionSensor;
		
		int nHorizontalPixels = camWidth - (camWidth % pixelsPerDirectionSensor);
		
		string log = "";
		for (int x = 0; x < nHorizontalPixels; x++) {
			Ray ray = agentCamera.ScreenPointToRay(new Vector3(x, horizonY, 0));
			
			RaycastHit[] castResult = Physics.RaycastAll(ray);
			
			if (castResult.Length != 0) {
				RaycastHit closestHit = GetClosestHit(castResult);
				RaycastHit wallHit = GetHitByTag(castResult, wallTag);
				int index = MiscUtils.IndexOf(recordCategoryTags, closestHit.transform.tag);
				int wallIndex = MiscUtils.IndexOf(recordCategoryTags, wallTag);
				if (index == -1) {
					Debug.LogWarning("RayCast found unlisted object.");
				} else {
					//uncompressedSignals[index, x] = 1.0f;
					
					log += Mathf.RoundToInt(closestHit.distance) + ";";
					
					if (!closestHit.transform.tag.Equals(wallTag)) {
						signals[index * nDirectionSensors + (x / pixelsPerDirectionSensor)] += pixelContribution;
						//Debug.DrawRay(ray.origin, 20.0f * ray.direction, Color.black, 10.0f, true);
					} 
				}              
				signals[wallIndex * nDirectionSensors + (x / pixelsPerDirectionSensor)] += pixelContribution / Mathf.Max(1.0f, wallHitMultiplier*wallHit.distance);                 
			}            
		}
		return signals;
	}


	private static RaycastHit GetClosestHit(RaycastHit[] castResult) {
		if (castResult.Length == 0) {
			throw new System.ArgumentException("castResult cannot be empty", "original");
		}
		
		float minDistance = float.PositiveInfinity;
		RaycastHit bestResult = castResult[0];
		foreach (RaycastHit result in castResult) {
			if (result.distance < minDistance) {
				minDistance = result.distance;
				bestResult = result;
			}
		}
		return bestResult;
	}


	private static RaycastHit GetHitByTag(RaycastHit[] castResult, string tag) {
		if (castResult.Length == 0) {
			throw new System.ArgumentException("castResult cannot be empty", "original");
		}
		
		foreach (RaycastHit result in castResult) {
			if (result.transform.tag.Equals(tag)) {
				return result;
			}
		}
		throw new System.ArgumentException("castResult does not contain tag", "original");
	}
}
