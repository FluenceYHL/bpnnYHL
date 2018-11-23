#include "gnuplot.hpp"
#include <random>
#include <assert.h>

namespace {
	using randType = std::uniform_real_distribution<double>;

	inline double getRand(const double edge){
		static std::default_random_engine e(time(0));
    	randType a(-edge, edge);
    	return a(e);
	};

	inline double sigmoid(const double x) {
		return 1.00 / (1.00 + std::exp(-x));
	}

	inline double dSigmoid(const double y) {
		return y * (1.00 - y);
	}
}

namespace YHL {

	class BPNN {
		static constexpr int layer = 3;
		static constexpr double N = 0.8;
		static constexpr double M = 0.2;
		using matrix = std::vector< std::vector<double> >;
	private:
		int iters = 7;
		double rate = 0.2;
		double init = 0.5;
		// layer - 1 层权值矩阵
		matrix weights[layer - 1];
		matrix before[layer - 1];
		// 数据集和答案, 如果需要存储,后期可以先存下来,多次训练
		matrix dataSet;
		matrix answers;
		// 两层之间的向量
		std::vector<double> output[layer];
		std::vector<double> delta[layer - 1];
		std::vector<double> threshold[layer - 1];

		int inputSize;
		int hideSize; 
		int outputSize;
		std::vector<double> target;

		int recognize() {
			this->forwardDrive();
			int best = -1;
			double max = -1e5;
			for(int i = 0;i < this->outputSize; ++i) {
				if(max < this->output[2][i]) {
					max = this->output[2][i];
					best = i;
				}
			}
			return best;
		}

		void initWeights() {
			for(int i = 0;i < inputSize; ++i) {
				this->weights[0].emplace_back(std::vector<double>());
				for(int j = 0;j < hideSize; ++j)
					this->weights[0][i].emplace_back(getRand(init));   // 这里也需要考虑
			}
			for(int i = 0;i < hideSize; ++i) {
				this->weights[1].emplace_back(std::vector<double>());
				for(int j = 0;j < outputSize; ++j)
					this->weights[1][i].emplace_back(getRand(init));
			}
		}

		void initOutput() {
			this->output[0].assign(inputSize, 0.00);
			this->output[1].assign(hideSize, 0.00);
			this->output[2].assign(outputSize, 0.00); 
		}

		void initDelta() {
			this->delta[0].assign(hideSize, 0.00);
			this->delta[1].assign(outputSize, 0.00);
		}

		void initThreshold() {
			for(int i = 0;i < hideSize; ++i)
				this->threshold[0].emplace_back(getRand(init));
			for(int i = 0;i < outputSize; ++i)
				this->threshold[1].emplace_back(getRand(init));
		}

		void initBefore() {
			// for(int i = 0;i < inputSize; ++i) 
			// 	this->before[0].emplace_back(std::vector<double>(hideSize, 0.00));
			// for(int i = 0;i < hideSize; ++i) 
			// 	this->before[1].emplace_back(std::vector<double>(outputSize, 0.00));
		}

		double getError() {
			double ans = 0;
			for(int i = 0;i < outputSize; ++i) 
				ans += std::pow(output[2][i] - target[i], 2);
			return ans * 0.50;
		}

		void forwardDrive() {
			for(int j = 0;j < hideSize; ++j) {
				double res = 0.00;
				for(int i = 0;i < inputSize; ++i)
					res += this->weights[0][i][j] * output[0][i];
				this->output[1][j] = sigmoid(res - this->threshold[0][j]);
			}
			for(int k = 0;k < outputSize; ++k) {
				double res = 0.00;
				for(int j = 0;j < hideSize; ++j)
					res += this->weights[1][j][k] * output[1][j];
				this->output[2][k] = sigmoid(res - this->threshold[1][k]);
			}
		}

		void backPropagate() {
			for(int i = 0;i < outputSize; ++i) {
				auto O = this->output[2][i];
				this->delta[1][i] = (O - target[i]) * dSigmoid(O);	
			}
			for(int k = 0;k < outputSize; ++k) {
				double gradient = delta[1][k];
				for(int j = 0;j < hideSize; ++j) {
					auto C = -rate * output[1][j] * gradient;
					this->weights[1][j][k] += C;
					// this->weights[1][j][k] += N * C + M * this->before[1][j][k];
					// this->before[1][j][k] = C;
				}
				this->threshold[1][k] -= rate * 1 * gradient;
			}
			for(int j = 0;j < hideSize; ++j) {
				auto res = 0.00;
				for(int k = 0;k < outputSize; ++k)  // 每个输出神经元的梯度, 和相联的边
					res += this->weights[1][j][k] * delta[1][k];
				auto O = this->output[1][j];
				this->delta[0][j] = dSigmoid(O) * res;
			}
			for(int j = 0;j < hideSize; ++j) {
				double gradient = delta[0][j];
				for(int i = 0;i < inputSize; ++i) {
					auto C = -rate * output[0][i] * gradient;
					this->weights[0][i][j] += C;
					// this->weights[0][i][j] += N * C + M * this->before[0][i][j];
					// this->before[0][i][j] = C;
				}
				this->threshold[0][j] -= rate * 1 * gradient;
			}
		}

		void initilize() {
			this->initWeights();
			this->initOutput();
			this->initDelta();
			this->initThreshold();
			this->initBefore();
		}

	public:
		BPNN(const int l, const int m, const int r) 
				: inputSize(l), hideSize(m), outputSize(r) {
			this->initilize();
		}
		BPNN(){}

		~BPNN() {
			std::ofstream out("weights2.txt", std::ios::trunc);
			ON_SCOPE_EXIT([&]{ 
				out.close(); 
			});
			assert(out);
			for(const auto& it : this->weights) {
				out << it.size() << "\n";
				for(const auto& l : it) {
					out << l.size() << "\n";
					for(const auto r : l)
						out << r << " ";
					out << "\n";
				}
			}
			std::cout << "矩阵备份完毕\n";
		}

		void train() {
			target.assign(outputSize, 0.00);
			for(int t = 0;t < iters; ++t) {
				std::cout << "开始读文件\n";
				std::ifstream in("./Minist_Rand.txt");
				ON_SCOPE_EXIT([&]{
					in.close();
				});
				for(int i = 0;i < 60000; ++i) {
					for(int j = 0;j < 784; ++j)
						in >> output[0][j];
					int ans;
					in >> ans;
					for(int i = 0;i < 10; ++i) 
						this->target[i] = 0.00;
					this->target[ans] = 1.00;
					this->forwardDrive();
					this->backPropagate();
				}
				std::cout << "第 " << t + 1 << " 轮训练隐含层层数结束...." << "\n";
			}
		}

		void load(const std::string& fileName) {
			std::ifstream in(fileName.c_str());
			ON_SCOPE_EXIT([&]{
				in.close();
			});
			in >> this->inputSize >> this->hideSize >> this->outputSize;
			in >> this->iters >> this->rate >> this->init;
			this->initilize();
			for(int i = 0;i < inputSize; ++i) {
				for(int j = 0;j < hideSize; ++j)
					in >> this->weights[0][i][j];
			}
			for(int i = 0;i < hideSize; ++i) {
				for(int j = 0;j < outputSize; ++j)
					in >> this->weights[1][i][j];
			}
			for(int i = 0;i < hideSize; ++i)
				in >> this->threshold[0][i];
			for(int i = 0;i < outputSize; ++i)
				in >> this->threshold[1][i];
		}

		int recognize(const std::vector<double>& input) {
			const int len = input.size();
			assert(len == inputSize);
			for(int i = 0;i < inputSize; ++i)
				this->output[0][i] = input[i];
			return this->recognize();
		}

		point test() {
			this->target.assign(this->outputSize, 0.00);
			std::ifstream in("./MinistTest.txt");
			ON_SCOPE_EXIT([&]{
				in.close();
			});
			assert(in);
			int cases, ans, cnt = 0;
			in >> cases;
			for(int t = 0;t < cases; ++t) {
				for(int i = 0;i < this->inputSize; ++i)
					in >> this->output[0][i];
				in >> ans;
				if(this->recognize() == ans) ++cnt;
			}
			std::cout << "成功率  :  " << cnt * 1.00 / cases << '\n';
			return point(this->rate, cnt * 1.00 / cases);
		}
	};

}

int main() {
	YHL::BPNN bpnn;
	bpnn.load("./bestBPNN.txt"); 
	bpnn.test();
	return 0;
}