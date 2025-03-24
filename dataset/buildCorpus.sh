echo "SCRAPING ..."

# Activate the virtual environment
source ../nlp_env/bin/activate

# Run the scraper for each game in a sub-process
(cd data
for seriesDir in ./*/
do
    if [[ ($seriesDir != "./Test/") &&  ($seriesDir != "./ALL/") ]]; then
        (cd "${seriesDir}"
        for gameDir in ./*/
        do
            (cd "${gameDir}"
            echo ${gameDir}
            mkdir -p raw
            python3 scraper.py)
        done)
    fi
done)

echo "PARSING ..."

(cd processing
python3 parseRawData.py
python3 getStatistics.py)