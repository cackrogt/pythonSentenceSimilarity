#include "pch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

using namespace std;

class tfidf {
  private:
    std::vector < std::string > patentsList; // all patents in order
  std::vector < std::vector < double >> dataMat; // converted bag of words matrix
  unsigned int nrow; // matrix row number
  unsigned int ncol; // matrix column number
  std::vector < std::vector < double >> weightMat; //tfidf weight matrix
  std::vector < std::vector < std::string >> rawDataSet; // raw data
  std::vector < std::string > vocabList; // all terms
  std::map < std::string, int > h_hot; // hot num
  std::vector < int > numOfTerms; // used in tf calculation
  std::vector < std::string > stopWords; // list of common words to ignore

  std::string readFileText(std::string & filename) {
    std::ifstream in (filename);
    std::string str((std::istreambuf_iterator < char > ( in )),
      std::istreambuf_iterator < char > ());
    return str;
  }

  void createVocabList() {
    std::set < std::string > vocabListSet;
    for (std::vector < std::string > document: rawDataSet) {
      for (std::string word: document)
        vocabListSet.insert(word);
    }
    std::copy(vocabListSet.begin(), vocabListSet.end(), std::back_inserter(vocabList));
  }

  std::vector < double > bagOfWords2VecMN(std::vector < std::string > & inputSet) {
    std::vector < double > returnVec(vocabList.size(), 0);
    for (std::string word: inputSet) {
      size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
      if (idx == vocabList.size())
        cout << "word: " << word << "not found" << endl;
      else
        returnVec.at(idx) += 1;
    }
    return returnVec;
  }

  void vec2mat() {
    cout << "Converting text to vector..." << endl;
    int cnt(0);
    for (auto it = rawDataSet.begin(); it != rawDataSet.end(); ++it) {
      cnt++;
      cout << cnt << "\r";
      std::cout.flush();
      dataMat.push_back(bagOfWords2VecMN( * it));
      numOfTerms.push_back(it -> size());
      it -> clear();
    }
    cout << endl;
    ncol = dataMat[0].size();
    nrow = dataMat.size();
    rawDataSet.clear(); // release memory
  }

  std::vector < std::string > textParse(std::string & bigString) {
    std::vector < std::string > vec;
    boost::tokenizer < > tok(bigString);
    for (boost::tokenizer < > ::iterator beg = tok.begin(); beg != tok.end(); ++beg) {
      if (!(std::binary_search(stopWords.begin(), stopWords.end(), * beg)))
        vec.push_back( * beg);
    }
    return vec;
  }

  std::vector < double > vec_sum(const std::vector < double > & a,
    const std::vector < double > & b) {
    assert(a.size() == b.size());
    std::vector < double > result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(),
      std::back_inserter(result), std::plus < double > ());
    return result;
  }

  // larger -> better
  double cosine_similarity(std::vector < double > & v1, std::vector < double > & v2) {
    unsigned int len = v1.size();
    double d1 = 0;
    double d2 = 0;
    double d3 = 0;
    for (unsigned int i = 0; i < len; ++i) {
      d1 += (v1[i] * v2[i]);
      d2 += (v1[i] * v1[i]);
      d3 += (v2[i] * v2[i]);
    }
    return d1 / (sqrt(d2) * sqrt(d3));
  }

  void orderByHot(std::vector < std::string > * vec) {
    std::map < int, std::vector < std::string >> temp;
    for (std::string e: * vec) {
      temp[h_hot[e]].push_back(e);
    }
    vec -> clear();
    for (auto it = temp.rbegin(); it != temp.rend(); ++it) {
      for (std::string e: it -> second)
        vec -> push_back(e);
    }
  }

  public:
    unsigned int recAmount;
  unsigned int finishCount;

  void loadData() {
    cout << "Loading data..." << endl;
    ifstream in ("patentList.csv");
    string tmp;
    std::vector < std::string > vec_str;
    int cnt = 0;
    while (! in .eof()) {
      cnt++;
      if (cnt > 500) // FOR TEST ONLY
        break;
      getline( in , tmp, '\n');
      if (tmp == "") break;
      boost::split(vec_str, tmp, boost::is_any_of(","));
      std::vector < std::string > wordList = textParse(vec_str[1]);
      rawDataSet.push_back(wordList);
      patentsList.push_back(vec_str[0]);
      tmp.clear();
      vec_str.clear();
    }

    std::ifstream in2("song_hot_num.csv");
    while (!in2.eof()) {
      getline(in2, tmp, '\n');
      if (tmp == "") break;
      boost::split(vec_str, tmp, boost::is_any_of(","));
      h_hot[vec_str[0]] = atoi(vec_str[1].c_str());
      tmp.clear();
      vec_str.clear();
    }
  }

  void loadStopWords() {
    ifstream in ("stop_words.txt");
    string tmp;
    while (! in .eof()) {
      getline( in , tmp, '\n');
      stopWords.push_back(tmp);
      tmp.clear();
    }
    std::sort(stopWords.begin(), stopWords.end());
  }

  void getMat() {
    cout << "Total " << rawDataSet.size() << " patents." << endl;
    cout << "Processing..." << endl;
    createVocabList();
    vec2mat();
    cout << "Calculating TF-IDF weight matrix..." << endl;
    std::vector < std::vector < double >> dataMat2(dataMat);
    std::vector < double > termCount;
    termCount.resize(ncol);

    for (unsigned int i = 0; i != nrow; ++i) {
      for (unsigned int j = 0; j != ncol; ++j) {
        if (dataMat2[i][j] > 1) // only keep 1 and 0
          dataMat2[i][j] = 1;
      }
      termCount = vec_sum(termCount, dataMat2[i]); // no. of doc. each term appears
      // from what we understand:
      // assume there are 5 words in total, that are not stop words
      // then dataMat has data as follows

      //       w1 , w2 , w3 , w4 , w5
      // p1    0  , 2 ,   1 ,  0 ,  1 
      // p2    1  , 1 ,   0 ,  1 ,  0 
      // p3    2  , 1 ,   1 ,  0 ,  0 
      // ... etc

      // similarly dataMat2 has the following:
      // 
      //       w1 , w2 , w3 , w4 , w5
      // p1    0  , 1 ,   1 ,  0 ,  1 
      // p2    1  , 1 ,   0 ,  1 ,  0 
      // p3    1  , 1 ,   1 ,  0 ,  0  // basically all 2s and greater ones are made 1s
      // ... etc

      // this is done as dataMat2 is used to find if a word exits in any patent abstract, which is a true/false check.

      // vec_sum is basically summing the values of each patent(exist(1)/absent(0)) for each word, and would be something like
      // 1st iter: 
      // 
      //       p1(is used)
      // w1     0
      // w2     1
      // w3     1
      // w4     0
      // w5     1
      // 2nd iter: 
      // 
      //       p2(is used)
      // w1     1
      // w2     2
      // w3     1
      // w4     1
      // w5     1
      // ... 
      // etc this is document frequency for each word basically. we will eventually use its reverse.

    }
    dataMat2.clear(); //release

    std::vector < double > row_vec;
    for (unsigned int i = 0; i != nrow; ++i) {
      cout << "\r" << (i + 1);
      std::cout.flush();
      for (unsigned int j = 0; j != ncol; ++j) {
        double tf = dataMat[i][j] / numOfTerms[i]; // word frequency in any doc / number of terms in that patent.
        double idf = log((double) nrow / (termCount[j])); // number of patent abstracts / termCounts for each word(as described above)
        row_vec.push_back(tf * idf); // TF-IDF equation (for each word(i, j, k unit vectors rep the words well), the value is tf*idf for one particular abstract)
      }
      weightMat.push_back(row_vec); // new matrix, contains the word based vectors for each abstract.
      row_vec.clear();
    }
    nrow = weightMat.size(); // ??? why is this needed.
    cout << endl;
  }

  void saveMat(std::string filename) {
    cout << "Saving weight matrix to " << filename << "..." << endl;
    std::ofstream outfile;
    outfile.open(filename, std::ios_base::app);
    for (auto it = weightMat.begin(); it != weightMat.end(); ++it) {
      std::ostringstream ss;
      for (auto it2 = it -> begin(); it2 != it -> end(); ++it2) {
        ss << * it2 << ",";
      }
      outfile << ss.str().substr(0, ss.str().size() - 1) << endl;
      ss.clear();
    }
  }

  void calSimi(unsigned int limit1, unsigned int limit2) {
    cout << "Calculating..." << endl;
    double similarity;
    std::map < double, std::vector < std::string >> similarPatent;
    std::vector < std::string > patent;
    for (std::size_t i = limit1; i != limit2; ++i) // for each patent
    {
      finishCount++;
      for (std::size_t j = 0; j != nrow; ++j) // compare with each patent
      {
        if (j != i) // exclude patent itself
        {
          similarity = cosine_similarity(weightMat[i], weightMat[j]);
          similarPatent[similarity].push_back(patentsList[j]);
        }
      }

      if (0 < similarPatent.rbegin() -> first) { // if the last one is 0, that means all values are 0, so no sentences have similarity.
        for (auto it2 = similarPatent.rbegin(); it2 != similarPatent.rend(); ++it2) {
          if (patent.size() >= recAmount)
            break;
          std::vector < std::string > temp(it2 -> second); // all patents that have a particular similarity
          orderByHot( & temp); // ?? what the f is this, this vector has a string something like: p1, p2, p3...
          for (std::string e: temp)
            patent.push_back(e); // each value in temp, we push it to patent. patent has all the values now, 
        }
        if (patent.size() >= recAmount) {
          std::vector < std::string > patent2(patent.begin(), patent.begin() + recAmount);
          std::ofstream outfile;
          outfile.open("similar_patents.txt", std::ios_base::app);
          outfile << patentsList[i] << "," << boost::join(patent2, ",") << endl; // we add 4 similar patents for patent i from patent list. 
          outfile.close();
          patent2.clear();
        } else {
          cout << "Unexpected small size recommendation list." << endl;
        }
      }
      patent.clear();
      similarPatent.clear();
    }
    cout << endl;
  }
};

int main() {
  tfidf patents;
  patents.loadStopWords();
  patents.loadData();
  patents.recAmount = 4;
  patents.getMat();
  patents.saveMat("tfidf_matrix.txt");
  patents.calSimi(0, 15);
}