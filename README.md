Step1 : docker pull qdrant/qdrant

Step2 : docker run -d -p 6333:6333 qdrant/qdrant
or 
If u want to store the data locally use the below command
docker run -d \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

