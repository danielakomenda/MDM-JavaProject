# Stage 1: Build the Svelte Frontend
FROM node:16 as frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ .
RUN npm run build


FROM maven:3.8.4-openjdk-17-slim as build
WORKDIR /usr/src/app
COPY . .
RUN mvn -Dmaven.test.skip=true package

FROM openjdk:17-jdk-slim
WORKDIR /usr/src/app
COPY --from=build /usr/src/app/target/javaproject-0.0.1-SNAPSHOT.jar .
COPY models /usr/src/app/models

EXPOSE 8081
CMD ["java","-jar","/usr/src/app/target/javaproject-0.0.1-SNAPSHOT.jar"]