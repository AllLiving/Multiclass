#include<iostream>
#include<bits\stdc++.h>
#include"lib\Comment.h"
#include"lib\Filestream.h"
#include"lib\Compute.h"
#include"lib\Neural.h"
using namespace std;

unsigned int hig = 0;
unsigned int mid = 0;
unsigned int low = 0;

unsigned int higShoot = 0;
unsigned int midShoot = 0;
unsigned int lowShoot = 0;

void InitKeyBase() {
	vector<string> keys;
	for (unsigned int kpin = 0; kpin < keybase.size(); kpin++) {
		if (keycnt[kpin] > mstFreq / 25) {
			keys.push_back(keybase[kpin]);
		}
	}
	keybase = keys;
	printf("[main]train:Select %d nodes as input datas.\n", keybase.size());
}

void valid(NeuNet &net, const vector<vector<double>> &matrix, const vector<double> &levels, const string &filename) {
	printf("[main]valid:Is predicting...\n");
	fstream fout(filename + ".csv", ios::out);
	for (unsigned int mpin = 0; mpin < matrix.size(); mpin++) {
		vector<double> crtv = matrix[mpin];
		double standard = levels[mpin];
		net.Forepass(crtv);
		double ans = net.Prophet();

		cout << mpin + 1 << ':' << ans << '\t' << forLevel(ans) << endl;

		if (standard < 0.1) {
			low++;
			if (forLevel(ans) == "LOW")
				lowShoot++;
		}
		else if (standard < 0.7) {
			mid++;
			if (forLevel(ans) == "MID")
				midShoot++;
		}
		else if (standard < 1.3) {
			hig++;
			if (forLevel(ans) == "HIG")
				higShoot++;
		}
		fout << forLevel(ans) << "\n";
	}

	printf("[main]:hig/HIG=%d/%d\n", higShoot, hig);
	printf("[main]:mid/MID=%d/%d\n", midShoot, mid);
	printf("[main]:low/LOW=%d/%d\n", lowShoot, low);
	printf("[main]valid:done.\n");
}

int main() {
	
	InitComma();

	NeuNet net;
	vector<double> levels;
	vector<vector<double>> matrix;
	printf("[main]train:Is reading file for training...\n");
	Filestream f("MulLabelTrain.ss");
	//Filestream f("MulLabelTest.ss");
	InitKeyBase();
	matrix = f.getCmtMtx();
	levels = f.getLevel();
	printf("[main]train:Read done.\n");

	Filestream ft("MulLabelTest.ss", false);
	vector<vector<double>> tmatrix = ft.getCmtMtx();
	vector<double> tlevels = ft.getLevel();

	Filestream fv("MulLabelValid.ss", false);
	vector<vector<double>> vmatrix = fv.getCmtMtx();
	vector<double> vlevels = fv.getLevel();

	net.Train(matrix, levels, 5, tmatrix, tlevels, vmatrix, vlevels);

	//printf("[main]valid:Is reading file for validing...\n");
	//Filestream fv("MulLabelTest.ss", false);
	//matrix = fv.getCmtMtx();
	//levels = fv.getLevel();
	//printf("[main]valid:Read done.\n");

	//valid(net, matrix, levels, "fout");

	
	printf("\n");	system("pause");/**/
	return 0;
}