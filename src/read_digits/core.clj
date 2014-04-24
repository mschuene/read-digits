(ns read-digits.core
  (:import [java.io RandomAccessFile]
           [java.awt.image BufferedImage]
           [javax.imageio ImageIO])
  (:require [clatrix.core :as c])
  (:use [clojure.core.matrix]
        [clojure.java.io :only [file]]))

(set-current-implementation :clatrix)

(def in (RandomAccessFile. "/home/kima/programming/digits/data0" "r"))

(defn read-pixel [in]
  (-> (.readByte in)
      (bit-and 0xFF)
      int))

(defn read-image [in]
  (matrix  (partition 28 (for [i (range (* 28 28))] (read-pixel in)))))

(defn write-image [image path]
  (let [bi (BufferedImage. 28 28 BufferedImage/TYPE_INT_RGB)]
    (doseq [y (range 28) x (range 28)]
      (let [value (mget image y x)
            rgb (+ value (bit-shift-left value 8) (bit-shift-left value 16))]
        (.setRGB bi x y rgb)))
    (ImageIO/write bi "PNG" (file path))))

(defn write-all-images []
  (doall (for [digit (range 10)
               :let [in (RandomAccessFile.
                         (str "/home/kima/programming/digits/data" digit) "r")]
               image (range 1000)]
           (write-image
            (read-image in)
            (str "/home/kima/programming/digits/" digit "s/" image ".png")))))

(defn my-as-vector [m]
  (matrix m (apply concat (slices m))))

(defn get-all-digits []
  (into {}
        (for [digit (range 1)
              :let [in (RandomAccessFile.
                        (str "/home/kima/programming/digits/data" digit) "r")]]
          [digit (mapv my-as-vector (for [i (range 1000)] (read-image in)))])))

(defn get-mean-vector [vectors]
  (div (reduce add vectors) (count vectors)))
;;jetzt nur erste 100
(defn create-mean-matrices  [numtraining]
  (for [i (range 0 10)
			:let [in (RandomAccessFile.
                        (str "/home/kima/programming/digits/data" i) "r")]]
    (as-> (for [i (range numtraining)] (read-image in)) x
          (reduce add x)
          (div x numtraining)
          (coerce :persistent-vector x)
          (spit
           (str "/home/kima/Dropbox/uni/Proseminar/read-digits/mean" i ".edn")
           x))))

(def mean-vectors
  (for [i (range 10)]
    (-> (str "/home/kima/Dropbox/uni/Proseminar/read-digits/mean" i ".edn")
        slurp
        read-string
        my-as-vector)))

(defn classify [mean-vectors to-classify]
  (->> (for [i (range 10)]
         [i (distance (nth mean-vectors i) to-classify)])
       (sort-by second)
       first
       first))

(defn run-svd-classifier [pc digit]
  (->> (map (fn [x r] [r (length (mmul x digit))]) pc (range))
       (sort-by second)
       first
       first))

(defn get-svd [digit num-training]
  (let [full-matrix (read-string (slurp (str (str "/home/kima/Dropbox/uni/Proseminar/read-digits/svd-" digit
                                                  "er-" num-training ".edn"))))]
    (c/svd (matrix :clatrix full-matrix))))

(defn train-svd-classifier [num-training basis-count]
  (let [us (map (comp :left #(get-svd % num-training)) (range 10))
        uks (map #(matrix :clatrix (select % :all (range basis-count))) us)
        id (identity-matrix 784)]
    (map #(sub id (mmul % (transpose %))) uks)))


;;95.11 Prozent Genauigkeit für alle Trainingsbilder
(defn classify-svd-test-results [pc num-training]
  (doall
   (for [digit (range 10)
         :let [in (RandomAccessFile.
                   (str "/home/kima/programming/digits/data" digit) "r")]]
     (do (prn "digit " digit)
         (dotimes [i num-training] (read-image in))
         (doall (for [i (range (- 1000 num-training))]
                  (do (prn "i " i)
                      {:actual digit :classified
                       (run-svd-classifier pc (my-as-vector (read-image in)))})))))))


(defn write-classify-svd-test-results []
  (let [res
        (doall
         (for [i (range 100 1000 100)]
           (let [_ (prn "calculating pc for " i)
                 pc (doall (train-svd-classifier i 10))
                 res (classify-svd-test-results pc i)]
             (spit (str "/home/kima/Dropbox/uni/Proseminar/read-digits/svd-res-" i
                        "-" 10 ".edn") (pr-str res))
             (count (filter #(= (:actual %) (:classified %)) (flatten res))))))]
    (spit "/home/kima/Dropbox/uni/Proseminar/read-digits/svd-res-summary-10"
          (pr-str res))
    res))

(defn classify-test-results [num-training]
  (let [_ (doall (create-mean-matrices num-training))
        mean-vectors
        (for [i (range 10)]
          (-> (str "/home/kima/Dropbox/uni/Proseminar/read-digits/mean" i ".edn")
              slurp
              read-string
              my-as-vector))]
    (doall (for [digit (range 10)
                 :let [in (RandomAccessFile.
                           (str "/home/kima/programming/digits/data" digit) "r")]]
             (do (prn "digit " digit)
                 (dotimes [i num-training] (read-image in))
                 (doall (for [i (range (- 1000 num-training))]
                          (do (prn "i " i)
                              {:actual digit :classified (classify mean-vectors
                                                                   (my-as-vector
                                                                    (read-image in)))}))))))))

;;77 prozent mit 100 test
;;nun finde alle prozentzahlen heraus
;;80.38 prozent wenn trainiert mit allen Daten trainiert

;;trainieren mit hundert und testen mit 900
;;hier sind die prozente
'(0.789375 0.7978571428571429 0.803 0.8038 0.79175 0.793 0.794 0.803)
(defn write-classify-results [anz-training-seq]
  (spit "/home/kima/Dropbox/uni/Proseminar/read-digits/prozents"
        (pr-str (doall
                 (for [anz-training anz-training-seq]
                   (let [results (classify-test-results anz-training)]
                     (spit
                      (str "/home/kima/Dropbox/uni/Proseminar/read-digits/classify-results"
                           anz-training ".edn")
                      (pr-str results))
                     [anz-training (double (/ (count (filter #(= (:actual %) (:classified %)) (flatten results))) (- 1000 anz-training)))]))))))


;;; First create the big training matrix for each digit/lets see if we can get all thousands

(defn linear-scale [values new-min new-max]
  (let [old-min (apply min values)
        old-max (apply max values)
        old-range (- old-max old-min)
        new-range (- new-max new-min)]
    ;;Prozentuell position auf old-range auf new-range projezieren
    (map #(let [proz (/ (- % old-min) old-range)]
            (+ new-min (* proz new-range))) values)))


(defn create-full-matrixes [num-training]
  (doall (for [digit (range 10)
               :let [in (RandomAccessFile.
                         (str "/home/kima/programming/digits/data" digit) "r")]]
           (->> (matrix :ndarray (for [i (range 1000)] (my-as-vector (read-image in))))
                transpose
                (coerce :persistent-vector)
                pr-str
                (spit (str "/home/kima/Dropbox/uni/Proseminar/read-digits/svd-" digit
                           "er-" num-training ".edn"))))))


(defn write-first-singular-pictures [num-training]
  (doall (for [digit (range 10)
               :let [svd (->> (str "/home/kima/Dropbox/uni/Proseminar/read-digits/svd-" digit
                                   "er-" num-training ".edn")
                              slurp
                              read-string
                              (matrix :clatrix)
                              c/svd)
                     u (:left svd)]]
           (doall (for [i (range 784)]
                    (as-> (get-column u i) x
                          (linear-scale x 0 255)
                          (map int x)
                          (partition 28 x)
                          (write-image x (str "/home/kima/programming/digits/singular-" digit "s/" num-training "/" i ".png"))))))))


(def pc (train-svd-classifier 1000 10))

