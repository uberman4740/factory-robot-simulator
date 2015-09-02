using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[ExecuteInEditMode]
public class LampPositioner : MonoBehaviour {
    public GameObject lampPrefab;

    public int nX;
    public int nZ;

    public float distX;
    public float distZ;
    public Vector3 corner;

    [SerializeField]
    private List<Object> instantiatedLamps = new List<Object>();
    public bool clearLamps;
    public bool createLamps;

    void Update() {
        if (createLamps 
            && lampPrefab != null
            && (instantiatedLamps == null
                || instantiatedLamps.Count == 0)) {

            instantiatedLamps = new List<Object>();
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nZ; j++) {
                    GameObject newLamp = (GameObject) Instantiate(lampPrefab,
                                        new Vector3(i * distX, 0, j * distZ) + corner,
                                        Quaternion.identity);
                    newLamp.name = "Lamp-" + i + "-" + j;
                    newLamp.transform.SetParent(this.transform);
                    instantiatedLamps.Add(newLamp);
                }
            }
            createLamps = false;
        }
        else if (createLamps) {
            Debug.Log("Lamps have already been created. Use Clear Lamps.");
            createLamps = false;
        }
        
        if (clearLamps) {
            foreach (Object lamp in instantiatedLamps) {
                DestroyImmediate(lamp);
            }
            instantiatedLamps.Clear();
            clearLamps = false;
        }
    }


}
