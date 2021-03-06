﻿using UnityEngine;
using System.Collections;
using System.Net;
using System.Net.Sockets;

public class InputListener : MonoBehaviour {

	public int port = 8889;
	private IPAddress ip = IPAddress.Any;

	private Socket socket;
	private byte[] buffer = new byte[16];

	private byte _currentFrameCounter;
	public byte currentFrameCounter {
		get {
			return _currentFrameCounter;
		}
	}
	private int _currentInputAction;
	public int currentInputAction {
		get {
			return _currentInputAction;
		}
	}
	private bool receivedBytes;


	void Start () {
		socket = new Socket(AddressFamily.InterNetwork, 
		                    SocketType.Dgram, 
		                    ProtocolType.Udp);

		socket.Bind(new IPEndPoint(ip, port));
		socket.Blocking = false;

		StartCoroutine(Poll());
	}

	IEnumerator Poll() {
		while (true) {
			yield return null;
			if (socket.Poll(0, SelectMode.SelectRead)) {
				int bytesReceived = socket.Receive(buffer, 
				                                   0, 
				                                   buffer.Length, 
				                                   SocketFlags.None);

				if (bytesReceived > 0) {
					receivedBytes = true;
					_currentFrameCounter = buffer[0];
					_currentInputAction = buffer[1];
				}
			}
		}
	}

	/** Queries the flag receivedBytes, which will be set to false. */
	public bool ReceivedBytes() {
		bool result = receivedBytes;
		receivedBytes = false;
		return result;
	}

	void OnApplicationQuit() {
		socket.Close();
	}


}
