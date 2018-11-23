#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <list>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <cstring>
#include <memory>
#include <assert.h>
#include <random>
#include <sys/unistd.h>
#include "scopeguard.h"

struct point {
	std::vector<double> features;
	point(const double x1, const double x2) {
		features.emplace_back(x1);
		features.emplace_back(x2);
	}
};

namespace YHL {

	class GNU_plot {
	private:
		FILE *pipeLine = nullptr;
		std::string curLine;
	public:
		GNU_plot() { pipeLine = popen("gnuplot", "w"); }
		~GNU_plot() {
			this->write("exit");
			fflush(pipeLine);
			if(pipeLine) {
				pclose(pipeLine);
				pipeLine = nullptr;
			}
		}
		void write(const char* cmd) {
			if(pipeLine and cmd and cmd[0]) 
				fprintf(this->pipeLine, "%s\n", cmd);
		}
		void write(const std::string& cmd) {
			if(!cmd.empty()) {
				this->write(cmd.c_str());
				fflush(pipeLine);
			}
		}
		void plot() {
			fflush(pipeLine);
			getchar();
		}
		void load(const std::vector<std::string>& container) {
			if(pipeLine == nullptr)
				return;
			for(const auto& it : container) 
				this->write(it.c_str());
		}
	};
}


namespace YHL {

	template<typename T>
	void writeFile(const T& oneSet, const std::string& fileName) {
		auto out = fopen(fileName.c_str(), "w+");
    	for(const auto& p : oneSet) {
    		for(const auto f : p.features) 
    			fprintf(out, "%lf ", f);
    		fprintf(out, "%s\n", "");
    	}
    	fclose(out);
	}

	template<typename T>
	void plotClusters(const T& clusters) {
		const int len = clusters.size();
	    std::vector<std::string> files;
	    for(int i = 0;i < len; ++i) {
	    	files.emplace_back("cluster(" + std::to_string(i) + ").txt");
		}

	    int cnt = 0;
	    for(const auto& it : clusters) 
	    	writeFile(it.second, files[cnt++]);

	    std::string commands = "plot ";
	    for(const auto& it : files) 
	    	commands += "'" + it + "', ";
	    commands.erase(commands.end() - 2, commands.end() - 1);
	    commands += " with points pointtype 5\n";  // with circles lc rgb 'yellow'

	    GNU_plot plotter;
	    plotter.write(commands);
	    plotter.plot();
	}

	void plot(const std::string& commands = "quit",
			  const std::string& title = "损失函数曲线",
			  const std::string& xlabelName = "迭代次数",
			  const std::string& ylabelName = "错误率") {
		GNU_plot plotter;
		plotter.write("set key outside");
		plotter.write("set title '" + title + "' font ',12'");
		plotter.write("set xlabel '" + xlabelName + "' font ',12'");
		plotter.write("set ylabel '" + ylabelName + "' font ',12'");   // set sample 50
		plotter.write(commands); 
		plotter.plot();
	}

	template<typename T>
	void plot(const T& curves, 
		      const std::string& fileName, 
		      const int font = 3, 
		      const std::string& title = "损失函数曲线",
			  const std::string& xlabelName = "迭代次数",
			  const std::string& ylabelName = "错误率"
			  ) {
		writeFile(curves, fileName);
		plot("plot '" + fileName + "' smooth csplines lw " + std::to_string(font),
			title, xlabelName, ylabelName);
	}

	template<typename T>
	void plots(const T& curves, 
		       const std::string& initName, 
		       const int font = 3,
		       const std::string& title = "损失函数曲线",
			   const std::string& xlabelName = "迭代次数",
			   const std::string& ylabelName = "错误率") {
		int cnt = -1;
		std::string commands = "plot ";
		for(const auto& it : curves) {
			auto fileName = initName;
			int pos = fileName.find('.');
			fileName.insert(pos, "(" + std::to_string(++cnt) + ")");
			writeFile(it, fileName);
			commands += "'" + fileName + "' smooth csplines lw " + std::to_string(font) + ", ";
		}
		commands[commands.size() - 2] = '\0';
		plot(commands, title, xlabelName, ylabelName);
	}

	// set label 'sin(x)' at 0.5,0.5 %在坐标（0.5,0.5）处加入字符串’sin(x)’。

	void plots(const std::vector<std::string>& fileNames = {
				"'curves.txt'", 
				"'curve(0).txt'",
				"'curve(1).txt'"
			   },
		       const std::vector<std::string>& curveNames = {
		       	"'正确率随着迭代次数的变化曲线'",
		       	"'正确率随着输入特征数目的变化曲线'",
		       	"'正确率随着学习率的变化曲线'"
		       },
		       const std::vector<std::string>& xlabelNames = {
		       	"'迭代次数'",
		       	"'特征维度'",
		       	"'学习率'"
		       },
		       const std::vector<std::string>& ylabelNames = {
		       	"'正确率'",
		       	"'正确率'",
		       	"'正确率'"
		       }
	) {
		YHL::GNU_plot plotter;
		plotter.write("set multiplot");
		plotter.write("set key outside");
		const int len = fileNames.size();
		std::cout << "len  :  " << len << "\n";
		int cnt = -1;
		for(const auto& it : fileNames) {
			plotter.write("set title " + curveNames[++cnt]);
			plotter.write("set xlabel " + xlabelNames[cnt]);
			plotter.write("set ylabel " + ylabelNames[cnt]);
			plotter.write("set origin 0.0, " + std::to_string(1./len));
			plotter.write("set size 1, " + std::to_string(1./len));
			plotter.write("plot " + it + " w l lw 3");
		}
		plotter.plot();
	}

}