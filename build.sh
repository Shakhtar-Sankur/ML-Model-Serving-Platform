#!/bin/bash
set -e

echo "Building ML Serving Platform..."

docker build -t ml-serving-api:latest .

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com

docker tag ml-serving-api:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com/ml-serving-api:latest

docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-west-2.amazonaws.com/ml-serving-api:latest

kubectl apply -f k8s/
kubectl rollout restart deployment/ml-serving-api -n ml-serving

echo "Deployment complete!"