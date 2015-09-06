using UnityEngine;
using System.Collections;
using System.IO;

public static class FileUtils {

    /** 
      * Taken from UntiyAnswers:
      * http://answers.unity3d.com/questions/245600/saving-a-png-image-to-hdd-in-standalone-build.html
      * */
    public static void SaveTextureToFile(Texture2D texture, string fileName) {
        byte[] bytes = texture.EncodeToPNG();
        //FileStream file = File.Open(Application.dataPath + "/" + fileName, FileMode.Create);
        FileStream file = File.Open(fileName, FileMode.Create);
        BinaryWriter writer = new BinaryWriter(file);
        writer.Write(bytes);
        file.Close();
    }

    public static void WriteStringToFile(string fileName, string content) {
        System.IO.File.WriteAllText(fileName, content);
    }

    public static void AppendStringToFile(string fileName, string content) {
        System.IO.File.AppendAllText(fileName, content);
    }

    public static void CopyFile(string from, string to) {
        if (System.IO.File.Exists(from)) {
            System.IO.File.Copy(from, to, true);
        }
    }
}
