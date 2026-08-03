#!/bin/bash
set -e

echo "Deploying to Kubernetes..."

kubectl create namespace ml-serving --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/pv-pvc.yaml

kubectl wait --for=condition=ready pod -l app=redis -n ml-serving --timeout=300s

kubectl apply -f k8s/triton-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/services.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

kubectl rollout status deployment/ml-serving-api -n ml-serving
kubectl rollout status deployment/triton-server -n ml-serving

echo "All services deployed successfully!"
kubectl get pods -n ml-serving