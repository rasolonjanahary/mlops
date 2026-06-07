pipeline {
    agent any

    stages {

        stage('Clone') {
            steps {
                git 'https://github.com/rasolonjanahary/mlops.git'
            }
        }

        stage('Préparation Python') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r require.txt
                '''
            }
        }

        stage('Analyse SonarQube') {
            steps {
                script {
                    def scannerHome = tool 'SonarScanner'

                    withSonarQubeEnv('SonarQube') {
                        sh """
                        ${scannerHome}/bin/sonar-scanner
                        """
                    }
                }
            }
        }

        stage('Entrainement') {
            steps {
                sh '''
                mkdir -p models artifacts
                . venv/bin/activate
                python src/train.py
                '''
            }
        }

        stage('Evaluation') {
            steps {
                sh '''
                . venv/bin/activate
                python src/evaluate.py
                '''
            }
        }

        stage('Tests') {
            steps {
                sh '''
                . venv/bin/activate
                python tests/test.py
                '''
            }
        }

        stage('Build Docker') {
            steps {
                sh '''
                docker build -t mlops-api:${BUILD_NUMBER} .
                '''
            }
        }

        stage('Push Harbor') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'harbor-credentials',
                        usernameVariable: 'HARBOR_USER',
                        passwordVariable: 'HARBOR_PASSWORD',
                        passwordVariable: 'HARBOR_URL'
                    )
                ]) {

                    sh '''
                    docker login ${HARBOR_URL} \
                    -u ${HARBOR_USER} \
                    -p ${HARBOR_PASSWORD}

                    docker tag mlops-api:${BUILD_NUMBER} \
                    ${HARBOR_URL}/diabete_detection/mlops-api:${BUILD_NUMBER}

                    docker push \
                    ${HARBOR_URL}/diabete_detection/mlops-api:${BUILD_NUMBER}
                    '''
                }
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
                 body: 'Déploiement terminé avec succès.'
        }

        failure {
            mail to: 'rasolonjanahary1263@gmail.com',
                 subject: 'Pipeline échouée',
                 body: 'Consultez les logs Jenkins.'
        }
    }
}