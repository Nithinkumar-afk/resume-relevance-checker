from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
import shutil, os, uuid

# ---------------------------
# Basic app + folders
# ---------------------------
app = FastAPI(title="Resume Analysis API (with JWT)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production: restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# Security / JWT config
# ---------------------------
SECRET_KEY = "change_this_to_a_random_long_secret_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# ---------------------------
# Fake user DB (temporary)
# ---------------------------
fake_users_db = {
    "recruiter@example.com": {
        "username": "recruiter@example.com",
        "full_name": "Recruiter One",
        "hashed_password": get_password_hash("test123"),
        "role": "recruiter"
    },
    "admin@example.com": {
        "username": "admin@example.com",
        "full_name": "Admin User",
        "hashed_password": get_password_hash("admin123"),
        "role": "admin"
    }
}

# ---------------------------
# Auth helper functions
# ---------------------------
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

# ---------------------------
# TokenData for JWT decode
# ---------------------------
class TokenData(BaseModel):
    username: str | None = None
    role: str | None = None

# ---------------------------
# JWT / Dependency Functions
# ---------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Provide a valid token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        # include full_name too
        user = fake_users_db.get(username)
        return {"username": username, "role": role, "full_name": user.get("full_name")}
    except JWTError:
        raise credentials_exception

async def get_current_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {"message": "Resume Analysis API (with JWT) is running."}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload_resume")
async def upload_resume(
    job_id: int = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analysis_result = {
        "resume_name": file.filename,
        "saved_path": file_path,
        "job_id": job_id,
        "uploaded_by": current_user["username"],
        "score": 72.5,
        "verdict": "Medium",
        "timestamp": datetime.utcnow().isoformat()
    }
    return {"analysis": analysis_result}

@app.get("/me")
async def read_current_user(current_user: dict = Depends(get_current_user)):
    return current_user

@app.delete("/admin/delete_resume/{filename}")
async def admin_delete_resume(filename: str, admin: dict = Depends(get_current_admin)):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"detail": f"Deleted {filename}"}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/protected")
def protected_route(current_user: dict = Depends(get_current_user)):
    return {
        "message": f"Hello {current_user['username']}! You are logged in as {current_user['role']}."
    }
