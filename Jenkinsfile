pipeline {
    agent any

    stages {
        stage("Clone"){
            steps {
                git "https://github.com/rasolonjanahary/mlops.git"
            }
        }

        stage("Preparation de l'environnement python"){
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r require.txt
                '''
            }
        }

        stage('Analyse de la vulanerabité sur le code') {
            steps {
                sh '''
                sonar-scanner
                '''
            }
        }

        stage('Entrainement du model') {
            steps {
                sh '''
                mkdir -p models artifacts
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Evaluation du model') {
            steps {
                sh '''
                . venv/bin/activate
                python evaluate.py
                '''
            }
        }

        stage('Tester le fonctionnement du model') {
            steps {
                sh '''
                . venv/bin/activate
                python register_model.py
                '''
            }
        }

        stage('Builder l\'image docker') {
            steps {
                sh '''
                docker build -t mlops-api:${BUILD_NUMBER} .
                '''
            }
        }

        stage('Push l\'image vers harbord') {
            steps {
                sh '''
                docker tag mlops-api:${BUILD_NUMBER} ${HARBORD_URL}/diabete_detection/mlops-api:${BUILD_NUMBER}
                docker push ${HARBORD_URL}/diabete_detection/mlops-api:${BUILD_NUMBER}
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                docker compose up -d
                '''
            }
        }
    }

    post {
        success {
            mail to: 'rasolonjanahary1263@gmail.com',
                subject: 'Pipeline réussi',
                body: 'Déploiement terminé'
        }

        failure {
            mail to: 'rasolonjanahary1263@gmail.com',
                subject: 'Pipeline échoué',
                body: 'Vérifiez Jenkins'
        }
    }
}