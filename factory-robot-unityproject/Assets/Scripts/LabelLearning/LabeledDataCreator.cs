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
        readerTexture = new Texture2D(cameraRenderTexture.width,
                                      cameraRenderTexture.height);


        FileUtils.CopyFile(trainingFilePath + labelFileName, trainingFilePath + labelFileName + "_backup");
        FileUtils.WriteStringToFile(trainingFilePath + labelFileName, "");
        
		nextSnapshotTime = Time.time + 5*snapshotPeriod;
    }

    void Update() {
        if (Time.time > nextSnapshotTime && counter < nCaptures) {
			nextSnapshotTime += snapshotPeriod;

			if (snapSwitch) {
				LabeledDataUtil.Snapshot(agentCamera,
				                         cameraRenderTexture,
				                         readerTexture,
				                         trainingFilePath,
				                         captureFilePrefix,
				                         labelFileName,
				                         recordCategoryTags,
				                         nDirectionSensors,
				                         wallTag,
				                         counter,
				                         wallHitMultiplier: 0.4f,
				                    	 storeImages: true);
				counter++;
				
				if (counter % 1000 == 0) {
					Debug.Log("counter: " + counter);
				}
			} else {
				SetRandomPositions();
			}
			snapSwitch = !snapSwitch;
        }
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
