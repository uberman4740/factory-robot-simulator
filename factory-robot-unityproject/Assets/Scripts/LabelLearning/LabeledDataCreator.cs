using UnityEngine;
using System.Collections;
using UnityStandardAssets.Characters.ThirdPerson;


public class LabeledDataCreator : MonoBehaviour {
    public RenderTexture cameraRenderTexture;
    private Texture2D readerTexture;
    public Camera agentCamera;

    public GameObject agent;

    public Transform[] classificationObjects;
    public string[] recordCategoryTags;

    // Probability with which each object is chosen to be displayed in a given capture
    public float pSelectObject = 0.3f;

    public Vector3 agentRange;
    public Vector3 buildingSize;

    public float relativeAngleRange = 180.0f;
    public float relativeDistanceMin = 1.0f;
    public float relativeDistanceMax = 10.0f;

    public int nCaptures = 50000;
    private int counter = 0;

    public string trainingFilePath;
    public string labelFileName = "labels.dat";
    public string captureFilePrefix = "capture";

    public int nDirectionSensors = 5;

    public string wallTag = "Building";

	public float snapshotPeriod = .1f;
	private float nextSnapshotTime;
	private bool snapSwitch;

    /** Initializiation */
    void Start() {
        Time.timeScale = 1.0f;
        readerTexture = new Texture2D(cameraRenderTexture.width,
                                      cameraRenderTexture.height);

        //InvokeRepeating("Snapshot", 0.01f, switchPeriod);


        FileUtils.CopyFile(trainingFilePath + labelFileName, trainingFilePath + labelFileName + "_backup");
        FileUtils.WriteStringToFile(trainingFilePath + labelFileName, "");


        
        //Camera.main.enabled = false;

		nextSnapshotTime = Time.time + 5*snapshotPeriod;
    }

    void Update() {

        if (Time.time > nextSnapshotTime && counter < nCaptures) {
			nextSnapshotTime += snapshotPeriod;

			if (snapSwitch) {
				Snapshot();
			} else {
				SetRandomPositions();
			}
			snapSwitch = !snapSwitch;
        }



        //if (Input.GetKeyDown(KeyCode.Space)) {
        //    float[] labelInfo = GetSensorInfo();
        //    for (int c = 0; c < recordCategoryTags.Length; c++) {
        //        string l = recordCategoryTags[c] + ": ";
        //        for (int s = 0; s < nDirectionSensors; s++) {
        //            l += labelInfo[c * nDirectionSensors + s] + "  ";
        //        }
        //        Debug.Log(l);
        //    }
        //}
    }

    /** Compute the fractional signal a sensor perceives, given its relative angle
     * from the target, the discount per sensor spacing distance and the sensor spacing distance */
    private static float GetSignalFraction(float targetAngle, float selfAngle, float angularDecay, float deltaAngle) {
        float dimlessDiff = (targetAngle - selfAngle) / deltaAngle;
        return Mathf.Exp(Mathf.Log(angularDecay) * dimlessDiff * dimlessDiff);
    }

    /** Compute the fractional signal due to the distnace from the sensor */
    private static float GetDistanceDiscount(float distance, float minDistance) {
        return Mathf.Min(1.0f, 1.0f / (distance - minDistance + 1.0f));
    }


    private void SetRandomPositions() {
        agent.transform.position = MathUtils.getUniformRandomVector3(
            new Vector3(0, agent.transform.position.y, 0),
            agentRange);

        agent.transform.eulerAngles = new Vector3(0, Random.Range(0.0f, 360.0f), 0);
//        agent.transform.eulerAngles = new Vector3(0, 90.0f, 0);
		
        for (int i = 0; i < classificationObjects.Length; i++) {
            Transform obj = classificationObjects[i];

            if (Random.Range(0.0f, 1.0f) > pSelectObject) {
                obj.gameObject.SetActive(false);
            } else {
                obj.gameObject.SetActive(true);
            }

            float relativeAngle = Random.Range(-relativeAngleRange / 2.0f, relativeAngleRange / 2.0f);
            float distance = Random.Range(relativeDistanceMin, relativeDistanceMax);
            float angle = relativeAngle + agent.transform.eulerAngles.y;

            SetRandomNavigationTarget(classificationObjects[i]);

            float objY = 0.0f;
            obj.position = agent.transform.position + distance * new Vector3(Mathf.Sin(Mathf.Deg2Rad * angle), objY, Mathf.Cos(Mathf.Deg2Rad * angle));

            obj.eulerAngles = new Vector3(0.0f, Random.Range(0.0f, 360.0f), 0.0f);
        }
    }

    private float[] GetSensorInfo() {
        //if (obj.gameObject.activeSelf) {

        //    for (int z = 0; z < nDirectionSensorsPerCategory; z++) {
        //        float sensorAngle = sensorsDeltaAngle * (z - (nDirectionSensorsPerCategory / 2.0f - 0.5f));
        //        float signal = GetSignalFraction(relativeAngle, sensorAngle, angularSignalDecay, sensorsDeltaAngle);
        //        signal = signal * GetDistanceDiscount(distance, relativeDistanceMin);
        //        sensorSignals[nDirectionSensorsPerCategory * j + z] += signal;
        //    }
        //}

        int camWidth = agentCamera.pixelWidth;
        int nCategories = recordCategoryTags.Length;
        float[] signals = new float[nDirectionSensors*nCategories];


        //float[,] uncompressedSignals = new float[nCategories, camWidth];


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
                signals[wallIndex * nDirectionSensors + (x / pixelsPerDirectionSensor)] += pixelContribution / Mathf.Max(1.0f, 0.4f*wallHit.distance);                 
            }            
        }
        return signals;
    }

    private RaycastHit GetClosestHit(RaycastHit[] castResult) {
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

    private RaycastHit GetHitByTag(RaycastHit[] castResult, string tag) {
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

    /** Set objects' positions randomly and save a snapshot + label */
    void Snapshot() {
        

        agentCamera.Render();

        float[] sensorData = GetSensorInfo();

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
        
        counter++;

        if (counter % 1000 == 0) {
            Debug.Log("counter: " + counter);
        }
    }


    private float GetWallDistance(Transform t, float sensorAngle) {
        RaycastHit[] hits;

        t.Rotate(new Vector3(0, sensorAngle, 0));
        Vector3 direction = t.forward;     
        hits = Physics.RaycastAll(t.position, direction);
        t.Rotate(new Vector3(0, -sensorAngle, 0));

        float distance = float.PositiveInfinity;
        foreach (var hit in hits) {
            if (hit.transform.root.tag.Equals("Building")) {
                distance = hit.distance;
            }
        }
        if (distance == float.PositiveInfinity) {
            Debug.LogWarning("No wall was being hit!");
        }
        return distance;
    }

    private void SetRandomNavigationTarget(Transform t) {
        AICharacterControl ctrl = t.GetComponent<AICharacterControl>();
        if (ctrl == null) {
            return;
        }
        else {
            if (ctrl.target == null) {
                GameObject target = new GameObject(t.gameObject.name + "Target");
                //target.transform.SetParent(t);
                ctrl.target = target.transform;
            }
            ctrl.target.position = MathUtils.getUniformRandomVector3(Vector3.zero, agentRange);
        }
    }
}
