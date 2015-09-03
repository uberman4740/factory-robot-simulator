using UnityEngine;
using System.Collections;
using UnityStandardAssets.Characters.ThirdPerson;

public class LabeledDataCreator : MonoBehaviour {
    public RenderTexture cameraRenderTexture;
    private Texture2D readerTexture;
    public Camera agentCamera;

    public GameObject agent;

    public Transform[] classificationObjects;

    public int[] distinctObjectStartIndices; // if you have classificationObjects = [A, A, A, B, C, C, D, D], this should be [3, 4, 6].

    // Probability with which each object is chosen to be displayed in a given capture
    public float pSelectObject = 0.3f;

    public Vector3 agentRange;
    public Vector3 buildingSize;

    public float relativeAngleRange = 180.0f;
    public float relativeDistanceMin = 1.0f;
    public float relativeDistanceMax = 10.0f;

    private int counter = 0;

    public float switchPeriod = 0.2f;

    public string trainingFilePath;
    public string labelFileName = "labels.dat";
    public string captureFilePrefix = "capture";

    public int nDirectionSensorsPerCategory = 5;
    public float sensorsDeltaAngle = 10.0f;

    /** Determines from what angle onwards objects are no longer counted toward label */
    public float cameraAngle = 70.0f;

    /** To what fraction of the maximum should the signal decay one delta-angle next to center */
    public float angularSignalDecay = 0.5f;

    /** Initializiation */
    void Start() {
        Time.timeScale = 1.0f;
        readerTexture = new Texture2D(cameraRenderTexture.width,
                                      cameraRenderTexture.height);

        InvokeRepeating("Snapshot", 0.01f, switchPeriod);

        FileUtils.WriteStringToFile(trainingFilePath + labelFileName, "");
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

    /** Set objects' positions randomly and save a snapshot + label */
    void Snapshot() {
        agent.transform.position = MathUtils.getUniformRandomVector3(
            new Vector3(0, agent.transform.position.y, 0),
            agentRange);

        agent.transform.eulerAngles = new Vector3(0, Random.Range(0.0f, 360.0f), 0);

        float[] sensorSignals = new float[nDirectionSensorsPerCategory * (distinctObjectStartIndices.Length + 2)];
        int j = 0;
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

            SetRandomTarget(classificationObjects[i]);

            //Debug.Log("agent angle " + agent.transform.eulerAngles.y);
            float objY = 0.0f;
            obj.position = agent.transform.position + distance * new Vector3(Mathf.Sin(Mathf.Deg2Rad * angle), objY, Mathf.Cos(Mathf.Deg2Rad * angle));


            obj.eulerAngles = new Vector3(0.0f, Random.Range(0.0f, 360.0f), 0.0f);

            // labels
            if ((j < distinctObjectStartIndices.Length) &&
                (distinctObjectStartIndices[j] == i)) {
                j++;
            }

            // Range check.
            if (obj.gameObject.activeSelf
                    && obj.position.x < buildingSize.x / 2
                    && obj.position.x > -buildingSize.x / 2
                    && obj.position.z < buildingSize.z / 2
                    && obj.position.z > -buildingSize.z / 2
                    && relativeAngle > -cameraAngle/2
                    && relativeAngle < cameraAngle/2) {

                for (int z = 0; z < nDirectionSensorsPerCategory; z++) {
                    float sensorAngle = sensorsDeltaAngle * (z - (nDirectionSensorsPerCategory/2.0f - 0.5f));               
                    float signal = GetSignalFraction(relativeAngle, sensorAngle, angularSignalDecay, sensorsDeltaAngle);
                    signal = signal * GetDistanceDiscount(distance, relativeDistanceMin);
                    sensorSignals[nDirectionSensorsPerCategory * j + z] += signal;
                }
            }
        }

        for (int z = 0; z < nDirectionSensorsPerCategory; z++) {
            float sensorAngle = sensorsDeltaAngle * (z - (nDirectionSensorsPerCategory/2.0f - 0.5f));
            float distance = GetWallDistance(agent.transform, sensorAngle);

            
            float distanceSignal = GetDistanceDiscount(distance, relativeDistanceMin);
            //Debug.LogFormat("z: {0}, distanceSignal: {1}", z, distanceSignal);
            sensorSignals[z + nDirectionSensorsPerCategory * (distinctObjectStartIndices.Length + 1)] = distanceSignal;
        }

        string labelVector = "";
        foreach (float a in sensorSignals) {
            labelVector += a.ToString();
            labelVector += ",";
        }
        //Debug.Log(labelVector);


        agentCamera.Render();
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

    private void SetRandomTarget(Transform t) {
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
